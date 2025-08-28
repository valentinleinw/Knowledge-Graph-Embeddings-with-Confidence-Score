import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from models import TransEUncertainty, DistMultUncertainty, ComplExUncertainty
from csvEditor import csvEditor
from pykeen.models import TransE, DistMult, ComplEx
from pykeen.evaluation import LCWAEvaluationLoop
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
import evaluator
import negative_sampling_creator
from sklearn.model_selection import train_test_split
import numpy as np


class KnowledgeGraphDataset(Dataset):
    def __init__(self, file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Mapping for entities and relations
        self.entities = list(set(df['head']).union(set(df['tail'])))
        self.relations = list(set(df['relation']))
        
        # Create mappings from entity/relation names to indices
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(self.entities)}
        self.relation_to_idx = {relation: idx for idx, relation in enumerate(self.relations)}
        
        # Convert the triples into indices
        self.triples = []
        for _, row in df.iterrows():
            head_idx = self.entity_to_idx[row['head']]
            tail_idx = self.entity_to_idx[row['tail']]
            relation_idx = self.relation_to_idx[row['relation']]
            confidence = row['confidence_score']
            self.triples.append((head_idx, relation_idx, tail_idx, confidence))
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        # Return the triple and the confidence score
        head, relation, tail, confidence = self.triples[idx]
        return head, relation, tail, confidence
   
def initialize(file_path, batch_size):
    dataset = KnowledgeGraphDataset(file_path)
    num_entities = len(dataset.entities)
    num_relations = len(dataset.relations)
    
    train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)  # 80% train, 20% test+val
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # Split temp into 50% validation, 50% test
        
    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return dataset, num_entities, num_relations, train_loader, val_loader, test_loader, train_data, val_data, test_data

def training_loop(models, train_loader, val_loader, test_loader, optimizers, loss_function,
                  num_epochs, num_entities, embedding_dim, batch_size, margin, file_path, result_file,
                  patience=10, delta=1e-4):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Pre-select loss function (avoid per-batch if/else)
    loss_fn_map = {
        "loss": lambda model, pos, neg, conf: model.loss(pos, neg, conf, margin),
        "objective": lambda model, pos, neg, conf: model.objective_function(pos, neg, conf),
        "softplus": lambda model, pos, neg, conf: model.softplus_loss(pos, neg, conf),
        "gaussian": lambda model, pos, _, conf: model.gaussian_nll_loss(pos, conf),
        "contrastive": lambda model, pos, neg, conf: model.contrastive_loss(pos, neg, margin, conf),
        "divergence": lambda model, pos, _, conf: model.kl_divergence_loss(pos, conf),
    }
    selected_loss = loss_fn_map[loss_function]
    
    # Training Loop with validation and early stopping
    for name, model in models.items():
        print(f"\nTraining {name}...")
        loss_model = 0
        best_val_mrr = float('-inf')
        epochs_no_improve = 0

        model.train()  # Set model to training mode
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                heads, relations, tails, confidences = batch
                heads = torch.tensor(heads, dtype=torch.long)
                relations = torch.tensor(relations, dtype=torch.long)
                tails = torch.tensor(tails, dtype=torch.long)
                confidences = torch.tensor(confidences, dtype=torch.float)

                # Generate negative samples
                neg_triples = negative_sampling_creator.negative_sampling(
                    list(zip(heads, relations, tails, confidences)), num_entities)
                neg_heads, neg_relations, neg_tails = zip(*neg_triples)
                neg_heads = torch.tensor(neg_heads, dtype=torch.long)
                neg_relations = torch.tensor(neg_relations, dtype=torch.long)
                neg_tails = torch.tensor(neg_tails, dtype=torch.long)

                # Compute loss and optimize
                optimizers[name].zero_grad()
                pos_triples = torch.stack([heads, relations, tails], dim=1)
                neg_triples = torch.stack([neg_heads, neg_relations, neg_tails], dim=1)
                loss = selected_loss(model, pos_triples, neg_triples, confidences)
                loss.backward()
                optimizers[name].step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

            # Validation evaluation (early stopping based on this)
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    if isinstance(model, ComplExUncertainty):
                        _, val_mrr, _, _, _ = evaluator.evaluate_complex(model, val_loader, device)
                    else:
                        _, val_mrr, _, _, _ = evaluator.evaluate(model, val_loader, device)

                    print(f"Validation MRR: {val_mrr:.4f}")

                    # Early Stopping Check
                    if val_mrr > best_val_mrr + delta:
                        best_val_mrr = val_mrr
                        epochs_no_improve = 0
                        best_model_state = model.state_dict()
                    else:
                        epochs_no_improve += 1
                        print(f"No improvement for {epochs_no_improve} epoch(s).")
                        if epochs_no_improve >= patience:
                            print(f"Early stopping triggered at epoch {epoch+1}")
                            model.load_state_dict(best_model_state)
                            break
            

        loss_model = avg_train_loss

        print(f"\nEvaluating {name} on test set...")
        if test_loader is not None:
            model.eval()
            if isinstance(model, ComplExUncertainty):
                mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5 = evaluator.evaluate_complex(model, test_loader, device)
            else:
                mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5 = evaluator.evaluate(model, test_loader, device)

            print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_10}")
                
            function_name = "train_and_evaluate" + "_" + loss_function

            csvEditor.write_results_to_csv(result_file, function_name, name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_10,
                                           file_path, loss_model, num_epochs, embedding_dim, batch_size, margin)

def training_loop_neg_confidences_cosukg(models, train_loader, val_loader, test_loader, optimizers,
                                        num_epochs, num_entities, embedding_dim, batch_size, margin, file_path, result_file,
                  patience=10, delta=1e-4):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        loss_model = 0
        best_val_mrr = float('-inf')
        epochs_no_improve = 0
        
        model.train()  # Set model to training mode
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                heads, relations, tails, confidences = batch
                heads = torch.tensor(heads, dtype=torch.long)
                relations = torch.tensor(relations, dtype=torch.long)
                tails = torch.tensor(tails, dtype=torch.long)
                pos_confidences = torch.tensor(confidences, dtype=torch.float)  # Renamed for clarity

                # Generate negative samples with confidence scores
                neg_quad = negative_sampling_creator.negative_sampling_cosukg(
                    list(zip(heads, relations, tails, pos_confidences)), num_entities, 10, x1=0.8, x2=0.2
                )

                # Unzip negative samples
                neg_heads, neg_relations, neg_tails, neg_confidences = zip(*neg_quad)
                neg_heads = torch.tensor(neg_heads, dtype=torch.long)
                neg_relations = torch.tensor(neg_relations, dtype=torch.long)
                neg_tails = torch.tensor(neg_tails, dtype=torch.long)
                neg_confidences = torch.tensor(neg_confidences, dtype=torch.float)  # Convert to tensor

                # Compute loss and optimize
                optimizers[name].zero_grad()
                pos_triples = torch.stack([heads, relations, tails], dim=1)
                neg_triples = torch.stack([neg_heads, neg_relations, neg_tails], dim=1)

                loss = model.loss_neg(pos_triples, neg_triples, pos_confidences, neg_confidences, margin)
                loss.backward()
                optimizers[name].step()

                total_loss += loss.item()
        
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation evaluation (early stopping based on this)
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    if isinstance(model, ComplExUncertainty):
                        _, val_mrr, _, _, _ = evaluator.evaluate_complex(model, val_loader, device)
                    else:
                        _, val_mrr, _, _, _ = evaluator.evaluate(model, val_loader, device)

                    print(f"Validation MRR: {val_mrr:.4f}")

                    # Early Stopping Check
                    if val_mrr > best_val_mrr + delta:
                        best_val_mrr = val_mrr
                        epochs_no_improve = 0
                        best_model_state = model.state_dict()
                    else:
                        epochs_no_improve += 1
                        print(f"No improvement for {epochs_no_improve} epoch(s).")
                        if epochs_no_improve >= patience:
                            print(f"Early stopping triggered at epoch {epoch+1}")
                            model.load_state_dict(best_model_state)
                            break
            

        loss_model = avg_train_loss
            
        if test_loader is not None:
            print(f"\nEvaluating {name} on test set...")
            if isinstance(model, ComplExUncertainty):  # Check if the model is ComplEx
                mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5 = evaluator.evaluate_complex(model, test_loader, device)
            else:
                mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5 = evaluator.evaluate(model, test_loader, device)
            
            # Print results
            print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_10}")
            
            # Log results to CSV
            csvEditor.write_results_to_csv(result_file, "train_and_evaluate_neg_confidences_cosukg", name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_10, file_path, loss_model, num_epochs, embedding_dim, batch_size, margin)

def training_loop_neg_confidences_inverse(models, train_loader, val_loader, test_loader, optimizers,
                                        num_epochs, num_entities, embedding_dim, batch_size, margin, file_path, result_file,
                                        patience=10, delta=1e-4):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        loss_model = 0
        best_val_mrr = float('-inf')
        epochs_no_improve = 0
        
        model.train()  # Set model to training mode
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                heads, relations, tails, confidences = batch
                heads = torch.tensor(heads, dtype=torch.long)
                relations = torch.tensor(relations, dtype=torch.long)
                tails = torch.tensor(tails, dtype=torch.long)
                pos_confidences = torch.tensor(confidences, dtype=torch.float)  # Renamed for clarity

                # Generate negative samples with confidence scores
                neg_quad = negative_sampling_creator.negative_sampling_inverse(
                    list(zip(heads, relations, tails, pos_confidences)), num_entities, 10
                )

                # Unzip negative samples
                neg_heads, neg_relations, neg_tails, neg_confidences = zip(*neg_quad)
                neg_heads = torch.tensor(neg_heads, dtype=torch.long)
                neg_relations = torch.tensor(neg_relations, dtype=torch.long)
                neg_tails = torch.tensor(neg_tails, dtype=torch.long)
                neg_confidences = torch.tensor(neg_confidences, dtype=torch.float)  # Convert to tensor

                # Compute loss and optimize
                optimizers[name].zero_grad()
                pos_triples = torch.stack([heads, relations, tails], dim=1)
                neg_triples = torch.stack([neg_heads, neg_relations, neg_tails], dim=1)

                loss = model.loss_neg(pos_triples, neg_triples, pos_confidences, neg_confidences, margin)
                loss.backward()
                optimizers[name].step()

                total_loss += loss.item()
        
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation evaluation (early stopping based on this)
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    if isinstance(model, ComplExUncertainty):
                        _, val_mrr, _, _, _ = evaluator.evaluate_complex(model, val_loader, device)
                    else:
                        _, val_mrr, _, _, _ = evaluator.evaluate(model, val_loader, device)

                    print(f"Validation MRR: {val_mrr:.4f}")

                    # Early Stopping Check
                    if val_mrr > best_val_mrr + delta:
                        best_val_mrr = val_mrr
                        epochs_no_improve = 0
                        best_model_state = model.state_dict()
                    else:
                        epochs_no_improve += 1
                        print(f"No improvement for {epochs_no_improve} epoch(s).")
                        if epochs_no_improve >= patience:
                            print(f"Early stopping triggered at epoch {epoch+1}")
                            model.load_state_dict(best_model_state)
                            break
            

        loss_model = avg_train_loss
    
        if test_loader is not None:
            print(f"\nEvaluating {name} on test set...")
            if isinstance(model, ComplExUncertainty):  # Check if the model is ComplEx
                mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5 = evaluator.evaluate_complex(model, test_loader, device)
            else:
                mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5 = evaluator.evaluate(model, test_loader, device)
            
            # Print results
            print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_10}")
            
            # Log results to CSV
            csvEditor.write_results_to_csv(result_file, "train_and_evaluate_neg_confidences_inverse", name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_10, file_path, loss_model, num_epochs, embedding_dim, batch_size, margin)

def training_loop_neg_confidences_similarity(models, train_loader, val_loader, test_loader, optimizers, 
                                            num_epochs, embedding_dim, batch_size, margin, file_path, result_file,
                                            patience=10, delta=1e-4, device="cuda"):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        loss_model = 0
        best_val_mrr = float('-inf')
        epochs_no_improve = 0

        model.to(device)
        model.train()

        for epoch in range(num_epochs):
            total_loss = 0

            # === Precompute neighbors once per epoch ===
            with torch.no_grad():
                if isinstance(model, ComplExUncertainty):
                    # ComplEx: concatenate real + imaginary embeddings
                    entity_embeddings = torch.cat(
                        [model.entity_im_embeddings.weight.detach(),
                        model.entity_re_embeddings.weight.detach()],
                        dim=1
                    )
                else:
                    entity_embeddings = model.entity_embeddings.weight.detach()

                # Move to CPU for sklearn
                entity_embeddings_cpu = entity_embeddings.cpu().numpy()

                # Compute top similar entities using sklearn
                top_similar, similarity_scores = negative_sampling_creator.precompute_similar_entities(
                    entity_embeddings_cpu, top_k=10
                )

                # Convert back to torch tensors on the original device
                top_similar = torch.tensor(top_similar, device=device, dtype=torch.long)
                similarity_scores = torch.tensor(similarity_scores, device=device, dtype=torch.float)


            for heads, relations, tails, confidences in train_loader:
                heads = heads.to(device, dtype=torch.long)
                relations = relations.to(device, dtype=torch.long)
                tails = tails.to(device, dtype=torch.long)
                confidences = confidences.to(device, dtype=torch.float32)
                pos_confidences = confidences  

                replace_mask = torch.rand(len(heads), device=device) > 0.5

                cand_heads = top_similar[heads]           
                prob_heads = similarity_scores[heads]    
                idx_heads = torch.multinomial(prob_heads, num_samples=1).squeeze()
                new_heads = cand_heads[torch.arange(len(heads), device=device), idx_heads]

                # Replace tails
                cand_tails = top_similar[tails]
                prob_tails = similarity_scores[tails]
                idx_tails = torch.multinomial(prob_tails, num_samples=1).squeeze()
                new_tails = cand_tails[torch.arange(len(tails), device=device), idx_tails]

                # Apply mask
                neg_heads = torch.where(replace_mask, new_heads, heads)
                neg_tails = torch.where(replace_mask, tails, new_tails)

                # Compute similarity scores for negatives
                sim_scores = torch.where(
                    replace_mask,
                    prob_heads[torch.arange(len(heads), device=device), idx_heads],
                    prob_tails[torch.arange(len(tails), device=device), idx_tails]
                )
                neg_confidences = pos_confidences * sim_scores

                optimizers[name].zero_grad()
                pos_triples = torch.stack([heads, relations, tails], dim=1)
                neg_triples = torch.stack([neg_heads, relations, neg_tails], dim=1)

                loss = model.loss_neg(pos_triples, neg_triples, pos_confidences, neg_confidences, margin)
                loss.backward()
                optimizers[name].step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}")

            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    if isinstance(model, ComplExUncertainty):
                        _, val_mrr, _, _, _ = evaluator.evaluate_complex(model, val_loader, device)
                    else:
                        _, val_mrr, _, _, _ = evaluator.evaluate(model, val_loader, device)

                print(f"Validation MRR: {val_mrr:.4f}")

                if val_mrr > best_val_mrr + delta:
                    best_val_mrr = val_mrr
                    epochs_no_improve = 0
                    best_model_state = model.state_dict()
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epoch(s).")
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        model.load_state_dict(best_model_state)
                        break
                model.train()

        loss_model = avg_train_loss

        if test_loader is not None:
            print(f"\nEvaluating {name} on test set...")
            if isinstance(model, ComplExUncertainty):
                mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5 = evaluator.evaluate_complex(model, test_loader, device)
            else:
                mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5 = evaluator.evaluate(model, test_loader, device)

            print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_10}")

            csvEditor.write_results_to_csv(
                result_file, "train_and_evaluate_neg_confidences_similarity", name,
                mean_rank, mrr, hits_at_1, hits_at_5, hits_at_10,
                file_path, loss_model, num_epochs, embedding_dim, batch_size, margin
            )

# helper
def evaluate_model_on_validation(model, val_loader):
    model.eval()  # Set model to evaluation mode
    
    mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5 = 0, 0, 0, 0, 0
    with torch.no_grad():  # Disable gradient calculation
        for batch in val_loader:
            heads, relations, tails, confidences = batch
            heads = torch.tensor(heads, dtype=torch.long)
            relations = torch.tensor(relations, dtype=torch.long)
            tails = torch.tensor(tails, dtype=torch.long)
            confidences = torch.tensor(confidences, dtype=torch.float)

            # Evaluate the model with the current batch
            if isinstance(model, ComplExUncertainty):  # Check if the model is ComplEx
                batch_mean_rank, batch_mrr, batch_hits_at_10, batch_hits_at_1, batch_hits_at_5 = evaluator.evaluate_complex(model, val_loader)
            else:
                # For other models, use the standard evaluate function
                batch_mean_rank, batch_mrr, batch_hits_at_10, batch_hits_at_1, batch_hits_at_5 = evaluator.evaluate(model, val_loader)
            mean_rank += batch_mean_rank
            mrr += batch_mrr
            hits_at_10 += batch_hits_at_10
            hits_at_1 += batch_hits_at_1
            hits_at_5 += batch_hits_at_5
    
    # Average the metrics over the validation set
    num_batches = len(val_loader)
    mean_rank /= num_batches
    mrr /= num_batches
    hits_at_10 /= num_batches
    hits_at_1 /= num_batches
    hits_at_5 /= num_batches
    
    return mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5
 
def train_and_evaluate_neg_confidences_cosukg(file_path, dataset_models, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    
    if embedding_dim % 2 != 0:
        raise ValueError("Embedding Dimensions have to be even")
    
    dataset, num_entities, num_relations, _, val_loader, _, train_data, val_data, test_data = initialize(file_path, batch_size)

    # Combine train and val for k-fold cross-validation
    train_val_data = train_data + val_data

    full_train_loader = DataLoader(train_val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    models = {
        "TransEUncertainty": TransEUncertainty(num_entities, num_relations, embedding_dim),
        "DistMultUncertainty": DistMultUncertainty(num_entities, num_relations, embedding_dim),
        "ComplExUncertainty": ComplExUncertainty(num_entities, num_relations, embedding_dim),
    }

    optimizers = {name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}

    # Final training and test evaluation, this time log to CSV
    training_loop_neg_confidences_cosukg(
        models, full_train_loader, val_loader=val_loader, test_loader=test_loader,
        optimizers=optimizers,
        num_epochs=num_epochs, num_entities=num_entities,
        embedding_dim=embedding_dim, batch_size=batch_size, margin=margin,
        file_path=file_path, result_file=result_file
    )

    # Optionally evaluate non-uncertainty models
    train_and_evaluate_normal_models(dataset_models, "train_and_evaluate_neg_confidences_cosukg", embedding_dim=embedding_dim, batch_size=batch_size, num_epochs=num_epochs, result_file=result_file)

def train_and_evaluate_neg_confidences_inverse(file_path, dataset_models, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    
    if embedding_dim % 2 != 0:
        raise ValueError("Embedding Dimensions have to be even")
    
    dataset, num_entities, num_relations, _, val_loader, _, train_data, val_data, test_data = initialize(file_path, batch_size)

    # Combine train and val for k-fold cross-validation
    train_val_data = train_data + val_data

    full_train_loader = DataLoader(train_val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    models = {
        "TransEUncertainty": TransEUncertainty(num_entities, num_relations, embedding_dim),
        "DistMultUncertainty": DistMultUncertainty(num_entities, num_relations, embedding_dim),
        "ComplExUncertainty": ComplExUncertainty(num_entities, num_relations, embedding_dim),
    }

    optimizers = {name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}

    # Final training and test evaluation, this time log to CSV
    training_loop_neg_confidences_inverse(
        models, full_train_loader, val_loader=val_loader, test_loader=test_loader,
        optimizers=optimizers,
        num_epochs=num_epochs, num_entities=num_entities,
        embedding_dim=embedding_dim, batch_size=batch_size, margin=margin,
        file_path=file_path, result_file=result_file
    )

    # Optionally evaluate non-uncertainty models
    train_and_evaluate_normal_models(dataset_models, "train_and_evaluate_neg_confidences_inverse", embedding_dim=embedding_dim, batch_size=batch_size, num_epochs=num_epochs, result_file=result_file)

def train_and_evaluate_neg_confidences_similarity(file_path, dataset_models, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv', k_folds=5):
    
    if embedding_dim % 2 != 0:
        raise ValueError("Embedding Dimensions have to be even")
    
    dataset, num_entities, num_relations, _, val_loader, _, train_data, val_data, test_data = initialize(file_path, batch_size)

    # Combine train and val for k-fold cross-validation
    train_val_data = train_data + val_data
    
    full_train_loader = DataLoader(train_val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    models = {
        "TransEUncertainty": TransEUncertainty(num_entities, num_relations, embedding_dim),
        "DistMultUncertainty": DistMultUncertainty(num_entities, num_relations, embedding_dim),
        "ComplExUncertainty": ComplExUncertainty(num_entities, num_relations, embedding_dim),
    }

    optimizers = {name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}

    # Final training and test evaluation, this time log to CSV
    training_loop_neg_confidences_similarity(
        models, full_train_loader, val_loader=val_loader, test_loader=test_loader,
        optimizers=optimizers,
        num_epochs=num_epochs,
        embedding_dim=embedding_dim, batch_size=batch_size, margin=margin,
        file_path=file_path, result_file=result_file
    )

    # Optionally evaluate non-uncertainty models
    train_and_evaluate_normal_models(dataset_models, "train_and_evaluate_neg_confidences_similarity", embedding_dim=embedding_dim, batch_size=batch_size, num_epochs=num_epochs, result_file=result_file)

def train_and_evaluate_normal_models(dataset, function_name, embedding_dim, batch_size, num_epochs, result_file='evaluation_results.csv'):
    dataset = dataset
    training = dataset.training
    validation = dataset.validation
    testing = dataset.testing
    assert validation is not None
    
    model = TransE(triples_factory=training, embedding_dim=embedding_dim)
    helper_for_normal_models(model, function_name, dataset.__class__.__name__, "TransE", num_epochs, batch_size, result_file, embedding_dim, training, validation, testing)
    
    model = DistMult(triples_factory=training, embedding_dim=embedding_dim)
    helper_for_normal_models(model, function_name, dataset.__class__.__name__, "DistMult", num_epochs, batch_size, result_file, embedding_dim, training, validation, testing)
    
    model = ComplEx(triples_factory=training, embedding_dim=embedding_dim)
    helper_for_normal_models(model, function_name, dataset.__class__.__name__, "ComplEx", num_epochs, batch_size, result_file, embedding_dim, training, validation, testing)
    
def helper_for_normal_models(model, function_name, dataset_name, name, num_epochs, batch_size, result_file, embedding_dim, training, validation, testing):

    model = model

    optimizer = Adam(params=model.get_grad_params())

    training_loop_local = SLCWATrainingLoop(
        model=model,
        triples_factory=training,
        optimizer=optimizer,
    )


    losses_per_epoch = training_loop_local.train(
        triples_factory=training,
        num_epochs=num_epochs,
        batch_size=batch_size,
        callbacks="evaluation-loop",
        callbacks_kwargs=dict(
            prefix="validation",
            factory=validation,
        ),
    )
    
    evaluation_loop = LCWAEvaluationLoop(
    model=model,
    triples_factory=testing,
)

    results = evaluation_loop.evaluate()
    
    mrr = results.get_metric('mean_reciprocal_rank')
    hits_at_1 = results.get_metric('hits@1')
    hits_at_5 = results.get_metric('hits@5')
    hits_at_10 = results.get_metric('hits@10')
    
    
    csvEditor.write_results_to_csv(result_file, function_name, name, "N/A", mrr, hits_at_1, hits_at_5, hits_at_10, dataset_name, losses_per_epoch[-1], num_epochs, embedding_dim, batch_size, "N/A")

def train_and_evaluate(file_path, dataset_models, loss_function="loss", embedding_dim=50, batch_size=64, 
                                num_epochs=10, margin=1.0, result_file='evaluation_results.csv',
                                patience = 10, delta=1e-4):
    dataset, num_entities, num_relations, _, val_loader, _, train_data, val_data, test_data = initialize(file_path, batch_size)
    
    if embedding_dim % 2 != 0:
        raise ValueError("Embedding Dimensions have to be even")

    # Combine train and val for k-fold cross-validation
    train_val_data = train_data + val_data

    full_train_loader = DataLoader(train_val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    models = {
        "TransEUncertainty": TransEUncertainty(num_entities, num_relations, embedding_dim),
        "DistMultUncertainty": DistMultUncertainty(num_entities, num_relations, embedding_dim),
        "ComplExUncertainty": ComplExUncertainty(num_entities, num_relations, embedding_dim),
    }

    optimizers = {name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}

    training_loop(
        models, full_train_loader, val_loader=val_loader, test_loader=test_loader,
        optimizers=optimizers, loss_function=loss_function,
        num_epochs=num_epochs, num_entities=num_entities,
        embedding_dim=embedding_dim, batch_size=batch_size, margin=margin,
        file_path=file_path, result_file=result_file, patience=patience, delta=delta
    )

    train_and_evaluate_normal_models(dataset_models, "train_and_evaluate" + "_" + loss_function, embedding_dim=embedding_dim, batch_size=batch_size, num_epochs=num_epochs, result_file=result_file)