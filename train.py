import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import TransEUncertainty, DistMultUncertainty, ComplExUncertainty, RotatEUncertainty
from csvEditor import csvEditor
from pykeen.models import TransE, DistMult, ComplEx, RotatE
from pykeen.evaluation import LCWAEvaluationLoop
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop




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


def negative_sampling(triples, num_entities):
    neg_triples = []
    for head, relation, tail, _ in triples:
        # Randomly corrupt head or tail
        if np.random.rand() > 0.5:
            new_head = np.random.randint(num_entities)
            neg_triples.append((new_head, relation, tail))
        else:
            new_tail = np.random.randint(num_entities)
            neg_triples.append((head, relation, new_tail))
    return neg_triples
    
def evaluate(model, dataset, top_k=10, device='cpu'):
    model.eval()
    
    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0
    
    entity_embeddings = model.entity_embeddings.weight  # Get all entity embeddings
    relation_embeddings = model.relation_embeddings.weight  # Get all relation embeddings
    
    entity_embeddings = entity_embeddings.to(device)
    relation_embeddings = relation_embeddings.to(device)

    for head, relation, tail, confidence in dataset:
        head_embedding = entity_embeddings[head].unsqueeze(0)  # (1, d)
        relation_embedding = relation_embeddings[relation].unsqueeze(0)  # (1, d)

        # Compute scores for all entities in parallel
        scores = torch.norm(head_embedding + relation_embedding - entity_embeddings, p=1, dim=1)  # (num_entities,)

        # Get the rank of the correct tail entity
        ranked_entities = torch.argsort(scores)  # Sort in ascending order
        rank = (ranked_entities == tail).nonzero(as_tuple=True)[0].item() + 1  # Convert to 1-based rank
        
        ranks.append((rank, confidence))
        
        # Check for Hits@1 and Hits@5
        hits_at_1 += (rank == 1)
        hits_at_5 += (rank <= 5)

    # Convert to numpy for faster operations
    ranks = np.array(ranks, dtype=np.float32)
    mean_rank = np.mean(ranks[:, 0])
    mrr = np.mean(1.0 / ranks[:, 0])
    hits_at_k = np.mean(ranks[:, 0] <= top_k)
    
    hits_at_1 = hits_at_1 / len(ranks)
    hits_at_5 = hits_at_5 / len(ranks)
    
    return mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5

# Evaluation for ComplEx model
def evaluate_complex(model, dataset, top_k=10):
    model.eval()
    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0

    entity_re_embeddings = model.entity_re_embeddings.weight.data
    entity_im_embeddings = model.entity_im_embeddings.weight.data
    relation_re_embeddings = model.relation_re_embeddings.weight.data
    relation_im_embeddings = model.relation_im_embeddings.weight.data

    for head, relation, tail, confidence in dataset:
        # Extract embeddings
        head_real, head_imag = entity_re_embeddings[head], entity_im_embeddings[head]
        relation_real, relation_imag = relation_re_embeddings[relation], relation_im_embeddings[relation]

        # Compute scores for all entities at once (vectorized)
        all_scores = torch.sum(
            (head_real * relation_real * entity_re_embeddings) +
            (head_imag * relation_real * entity_im_embeddings) +
            (head_real * relation_imag * entity_im_embeddings) -
            (head_imag * relation_imag * entity_re_embeddings),
            dim=1
        )

        # Rank entities efficiently using PyTorch
        sorted_indices = torch.argsort(all_scores, descending=True)
        rank = (sorted_indices == tail).nonzero(as_tuple=True)[0].item() + 1  # Convert to 1-based rank

        ranks.append((rank, confidence))

        # Compute Hits@1 and Hits@5
        hits_at_1 += (rank == 1)
        hits_at_5 += (rank <= 5)

    # Compute Evaluation Metrics
    mean_rank = np.mean([rank for rank, _ in ranks])
    mrr = np.mean([1 / rank for rank, _ in ranks])
    hits_at_k = np.mean([1 if rank <= top_k else 0 for rank, _ in ranks])

    # Normalize Hits@1 and Hits@5
    hits_at_1 /= len(ranks)
    hits_at_5 /= len(ranks)

    return mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5

def evaluate_rotate(model, dataset, top_k=10):
    model.eval()
    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0

    # Extract entity and relation embeddings
    entity_embeddings = model.entity_embeddings.weight.data  # (num_entities, embedding_dim)
    relation_embeddings = model.relation_embeddings.weight.data  # (num_relations, embedding_dim // 2)

    for head, relation, tail, confidence in dataset:
        # Extract embeddings
        head_embedding = entity_embeddings[head]
        tail_embedding = entity_embeddings[tail]
        relation_embedding = relation_embeddings[relation]  # Rotation in radians

        # Split into real and imaginary parts
        head_real, head_imag = torch.chunk(head_embedding, 2, dim=0)
        tail_real, tail_imag = torch.chunk(tail_embedding, 2, dim=0)

        # Compute rotation using cosine and sine
        cos_r = torch.cos(relation_embedding)
        sin_r = torch.sin(relation_embedding)

        rotated_head_real = head_real * cos_r - head_imag * sin_r
        rotated_head_imag = head_real * sin_r + head_imag * cos_r

        # Compute L2 distances to all entities (vectorized)
        all_real, all_imag = torch.chunk(entity_embeddings, 2, dim=1)
        all_scores = torch.norm(rotated_head_real - all_real, dim=1) + torch.norm(rotated_head_imag - all_imag, dim=1)

        # Rank entities efficiently
        rank = torch.argsort(all_scores).tolist().index(tail) + 1  # Convert to 1-based ranking

        ranks.append((rank, confidence))

        # Compute Hits@1 and Hits@5
        hits_at_1 += (rank == 1)
        hits_at_5 += (rank <= 5)

    # Compute Evaluation Metrics
    mean_rank = torch.tensor([rank for rank, _ in ranks], dtype=torch.float32).mean().item()
    mrr = torch.tensor([1 / rank for rank, _ in ranks], dtype=torch.float32).mean().item()
    hits_at_k = torch.tensor([1 if rank <= top_k else 0 for rank, _ in ranks], dtype=torch.float32).mean().item()

    # Normalize Hits@1 and Hits@5
    hits_at_1 /= len(ranks)
    hits_at_5 /= len(ranks)

    return mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5

def train_and_evaluate(file_path, dataset_models, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    # Load the dataset
    dataset = KnowledgeGraphDataset(file_path)
    num_entities = len(dataset.entities)
    num_relations = len(dataset.relations)
    
    # Initialize the model
    models = {
        "TransEUncertainty": TransEUncertainty(num_entities, num_relations, embedding_dim),
        "DistMultUncertainty": DistMultUncertainty(num_entities, num_relations, embedding_dim),
        "ComplExUncertainty": ComplExUncertainty(num_entities, num_relations, embedding_dim),
        "RotatEUncertainty": RotatEUncertainty(num_entities, num_relations, embedding_dim)
    }
    optimizers = {name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}
    
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training Loop
    for name, model in models.items():
        print(f"\nTraining {name}...")
        loss_model = 0
        
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
                neg_triples = negative_sampling(list(zip(heads, relations, tails, confidences)), num_entities)
                neg_heads, neg_relations, neg_tails = zip(*neg_triples)
                neg_heads = torch.tensor(neg_heads, dtype=torch.long)
                neg_relations = torch.tensor(neg_relations, dtype=torch.long)
                neg_tails = torch.tensor(neg_tails, dtype=torch.long)

                # Compute loss and optimize
                optimizers[name].zero_grad()
                pos_triples = torch.stack([heads, relations, tails], dim=1)
                neg_triples = torch.stack([neg_heads, neg_relations, neg_tails], dim=1)
                loss = model.loss(pos_triples, neg_triples, confidences, margin)
                loss.backward()
                optimizers[name].step()

                total_loss += loss.item()
        
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
        
        loss_model = total_loss / len(train_loader)
    
        print(f"\nEvaluating {name}...")
        if name == "ComplExUncertainty":
            mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5 = evaluate_complex(models[name], dataset)
        elif name == "RotatEUncertainty": 
            mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5 = evaluate_rotate(models[name], dataset)
        else:
            mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5 = evaluate(models[name], dataset)

        # Print results
        print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_k}")
        
        csvEditor.write_results_to_csv(result_file, name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k, file_path, loss_model, num_epochs, embedding_dim, batch_size, margin)
        
    train_and_evaluate_normal_models(dataset_models, embedding_dim, batch_size, num_epochs, margin, result_file=result_file)
    
def train_and_evaluate_normal_models(dataset, embedding_dim, batch_size, num_epochs, margin, result_file='evaluation_results.csv'):
    dataset = dataset
    training = dataset.training
    validation = dataset.validation
    testing = dataset.testing
    assert validation is not None
    
    model = TransE(triples_factory=training, embedding_dim=embedding_dim)
    helper_for_normal_models(model, dataset.__class__.__name__, "TransE", num_epochs, batch_size, result_file, embedding_dim, training, validation, testing)
    
    model = DistMult(triples_factory=training, embedding_dim=embedding_dim)
    helper_for_normal_models(model,dataset.__class__.__name__, "DistMult", num_epochs, batch_size, result_file, embedding_dim, training, validation, testing)
    
    model = ComplEx(triples_factory=training, embedding_dim=embedding_dim)
    helper_for_normal_models(model,dataset.__class__.__name__, "ComplEx", num_epochs, batch_size, result_file, embedding_dim, training, validation, testing)
    
    model = RotatE(triples_factory=training, embedding_dim=embedding_dim)
    helper_for_normal_models(model,dataset.__class__.__name__, "RotatE", num_epochs, batch_size, result_file, embedding_dim, training, validation, testing)

def helper_for_normal_models(model, dataset_name, name, num_epochs, batch_size, result_file, embedding_dim, training, validation, testing):

    model = model

    optimizer = Adam(params=model.get_grad_params())

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training,
        optimizer=optimizer,
    )


    losses_per_epoch = training_loop.train(
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
    
    
    csvEditor.write_results_to_csv(result_file, name, "N/A", mrr, hits_at_1, hits_at_5, hits_at_10, dataset_name, losses_per_epoch[-1], num_epochs, embedding_dim, batch_size, "N/A")
    
def train_and_evaluate_objective_function(file_path, dataset_models, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    # Load the dataset
    dataset = KnowledgeGraphDataset(file_path)
    num_entities = len(dataset.entities)
    num_relations = len(dataset.relations)
    
    # Initialize the model
    models = {
        "TransEUncertainty": TransEUncertainty(num_entities, num_relations, embedding_dim),
        "DistMultUncertainty": DistMultUncertainty(num_entities, num_relations, embedding_dim),
        "ComplExUncertainty": ComplExUncertainty(num_entities, num_relations, embedding_dim),
    }
    optimizers = {name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}
    
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        loss_model = 0

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
                neg_triples = negative_sampling(list(zip(heads, relations, tails, confidences)), num_entities)
                neg_heads, neg_relations, neg_tails = zip(*neg_triples)
                neg_heads = torch.tensor(neg_heads, dtype=torch.long)
                neg_relations = torch.tensor(neg_relations, dtype=torch.long)
                neg_tails = torch.tensor(neg_tails, dtype=torch.long)

                # Stack to create tensor batches
                pos_triples = torch.stack([heads, relations, tails], dim=1)
                neg_triples = torch.stack([neg_heads, neg_relations, neg_tails], dim=1)

                # Compute the loss using the objective function
                loss = model.objective_function(pos_triples, neg_triples, confidences)

                # Optimize
                optimizers[name].zero_grad()
                loss.backward()
                optimizers[name].step()

                total_loss += loss.item()
        
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
        
        loss_model = total_loss / len(train_loader)
        
        print(f"\nEvaluating {name}...")
        if name == "ComplExUncertainty":
            mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5 = evaluate_complex(models[name], dataset)
        elif name == "RotatEUncertainty": 
            mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5 = evaluate_rotate(models[name], dataset)
        else:
            mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5 = evaluate(models[name], dataset)

        # Print results
        print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_k}")
        
        csvEditor.write_results_to_csv(result_file, name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k, file_path, loss_model, num_epochs, embedding_dim, batch_size, margin)
        
    train_and_evaluate_normal_models(dataset_models, embedding_dim, batch_size, num_epochs, margin, result_file=result_file)
    
def train_transE_with_different_losses(file_path, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = KnowledgeGraphDataset(file_path)
    num_entities = len(dataset.entities)
    num_relations = len(dataset.relations)
    
    # Initialize the model
    model = TransEUncertainty(num_entities, num_relations, embedding_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_functions = {
        "softplus": model.softplus_loss,
        "gaussian": model.gaussian_nll_loss,
        "contrastive": model.contrastive_loss,
        "divergence": model.kl_divergence_loss,
        "objective_function": model.objective_function
    }
    
    for name, loss_function in loss_functions.items():
        print(f"\nTraining {name}...")
        
        model.train()  # Set model to training mode
        total_loss = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in train_loader:
                heads, relations, tails, confidences = [x.to(device) for x in batch]
                
                # Generate negative samples
                neg_triples = negative_sampling(list(zip(heads.cpu(), relations.cpu(), tails.cpu(), confidences.cpu())), num_entities)
                neg_heads, neg_relations, neg_tails = zip(*neg_triples)
                
                neg_heads = torch.tensor(neg_heads, dtype=torch.long, device=device)
                neg_relations = torch.tensor(neg_relations, dtype=torch.long, device=device)
                neg_tails = torch.tensor(neg_tails, dtype=torch.long, device=device)

                # Compute loss and optimize
                optimizer.zero_grad()
                pos_triples = torch.stack([heads, relations, tails], dim=1)
                neg_triples = torch.stack([neg_heads, neg_relations, neg_tails], dim=1)
                
                # Compute loss dynamically based on function selection
                if name == "contrastive":
                    loss = loss_function(pos_triples, neg_triples, margin)
                elif name in ["objective_function", "softplus"]:
                    loss = loss_function(pos_triples, neg_triples, confidences)
                else:
                    loss = loss_function(pos_triples, confidences)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")
            total_loss += avg_epoch_loss  # Accumulate total loss properly

        # Evaluation
        mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5 = evaluate(model, dataset)

        # Print results
        print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_k}")

        # Write to CSV
        csvEditor.write_results_to_csv(
            result_file, name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k,
            file_path, total_loss, num_epochs, embedding_dim, batch_size, margin
        )
        
def train_distmult_with_different_losses(file_path, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = KnowledgeGraphDataset(file_path)
    num_entities = len(dataset.entities)
    num_relations = len(dataset.relations)
    
    # Initialize the model
    model = DistMultUncertainty(num_entities, num_relations, embedding_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_functions = {
        "softplus": model.softplus_loss,
        "gaussian": model.gaussian_nll_loss,
        "contrastive": model.contrastive_loss,
        "divergence": model.kl_divergence_loss,
        "objective_function": model.objective_function
    }
    
    for name, loss_function in loss_functions.items():
        print(f"\nTraining {name}...")
        
        model.train()  # Set model to training mode
        total_loss = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in train_loader:
                heads, relations, tails, confidences = [x.to(device) for x in batch]
                
                # Generate negative samples
                neg_triples = negative_sampling(list(zip(heads.cpu(), relations.cpu(), tails.cpu(), confidences.cpu())), num_entities)
                neg_heads, neg_relations, neg_tails = zip(*neg_triples)
                
                neg_heads = torch.tensor(neg_heads, dtype=torch.long, device=device)
                neg_relations = torch.tensor(neg_relations, dtype=torch.long, device=device)
                neg_tails = torch.tensor(neg_tails, dtype=torch.long, device=device)

                # Compute loss and optimize
                optimizer.zero_grad()
                pos_triples = torch.stack([heads, relations, tails], dim=1)
                neg_triples = torch.stack([neg_heads, neg_relations, neg_tails], dim=1)
                
                # Compute loss dynamically based on function selection
                if name == "contrastive":
                    loss = loss_function(pos_triples, neg_triples, margin)
                elif name in ["objective_function", "softplus"]:
                    loss = loss_function(pos_triples, neg_triples, confidences)
                else:
                    loss = loss_function(pos_triples, confidences)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")
            total_loss += avg_epoch_loss  # Accumulate total loss properly

        # Evaluation
        mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5 = evaluate(model, dataset)

        # Print results
        print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_k}")

        # Write to CSV
        csvEditor.write_results_to_csv(
            result_file, name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k,
            file_path, total_loss, num_epochs, embedding_dim, batch_size, margin
        )
        
def train_complex_with_different_losses(file_path, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = KnowledgeGraphDataset(file_path)
    num_entities = len(dataset.entities)
    num_relations = len(dataset.relations)
    
    # Initialize the model
    model = ComplExUncertainty(num_entities, num_relations, embedding_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_functions = {
        "softplus": model.softplus_loss,
        "gaussian": model.gaussian_nll_loss,
        "contrastive": model.contrastive_loss,
        "divergence": model.kl_divergence_loss,
        "objective_function": model.objective_function
    }
    
    for name, loss_function in loss_functions.items():
        print(f"\nTraining {name}...")
        
        model.train()  # Set model to training mode
        total_loss = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in train_loader:
                heads, relations, tails, confidences = [x.to(device) for x in batch]
                
                # Generate negative samples
                neg_triples = negative_sampling(list(zip(heads.cpu(), relations.cpu(), tails.cpu(), confidences.cpu())), num_entities)
                neg_heads, neg_relations, neg_tails = zip(*neg_triples)
                
                neg_heads = torch.tensor(neg_heads, dtype=torch.long, device=device)
                neg_relations = torch.tensor(neg_relations, dtype=torch.long, device=device)
                neg_tails = torch.tensor(neg_tails, dtype=torch.long, device=device)

                # Compute loss and optimize
                optimizer.zero_grad()
                pos_triples = torch.stack([heads, relations, tails], dim=1)
                neg_triples = torch.stack([neg_heads, neg_relations, neg_tails], dim=1)
                
                # Compute loss dynamically based on function selection
                if name == "contrastive":
                    loss = loss_function(pos_triples, neg_triples, margin)
                elif name in ["objective_function", "softplus"]:
                    loss = loss_function(pos_triples, neg_triples, confidences)
                else:
                    loss = loss_function(pos_triples, confidences)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")
            total_loss += avg_epoch_loss  # Accumulate total loss properly

        # Evaluation
        mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5 = evaluate_complex(model, dataset)

        # Print results
        print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_k}")

        # Write to CSV
        csvEditor.write_results_to_csv(
            result_file, name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k,
            file_path, total_loss, num_epochs, embedding_dim, batch_size, margin
        )