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
import evaluator
import negative_sampling_creator

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
        
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataset, num_entities, num_relations, train_loader

def training_loop(models, train_loader, optimizers, loss_function, dataset, num_epochs, num_entities, embedding_dim, batch_size, margin, file_path, result_file):
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
                neg_triples = negative_sampling_creator.negative_sampling(list(zip(heads, relations, tails, confidences)), num_entities)
                neg_heads, neg_relations, neg_tails = zip(*neg_triples)
                neg_heads = torch.tensor(neg_heads, dtype=torch.long)
                neg_relations = torch.tensor(neg_relations, dtype=torch.long)
                neg_tails = torch.tensor(neg_tails, dtype=torch.long)

                # Compute loss and optimize
                optimizers[name].zero_grad()
                pos_triples = torch.stack([heads, relations, tails], dim=1)
                neg_triples = torch.stack([neg_heads, neg_relations, neg_tails], dim=1)
                if loss_function == "loss":
                    loss = model.loss(pos_triples, neg_triples, confidences, margin)
                else:
                    loss = model.objective_function(pos_triples, neg_triples, confidences)
                loss.backward()
                optimizers[name].step()

                total_loss += loss.item()
        
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
        
        loss_model = total_loss / len(train_loader)
    
        print(f"\nEvaluating {name}...")
        if name == "ComplExUncertainty":
            mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5 = evaluator.evaluate_complex(models[name], dataset)
        elif name == "RotatEUncertainty": 
            mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5 = evaluator.evaluate_rotate(models[name], dataset)
        else:
            mean_rank, mrr, hits_at_10, hits_at_1, hits_at_5 = evaluator.evaluate(models[name], dataset)

        # Print results
        print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_10}")
        
        csvEditor.write_results_to_csv(result_file, name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_10, file_path, loss_model, num_epochs, embedding_dim, batch_size, margin)

def train_and_evaluate(file_path, dataset_models, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    
    dataset, num_entities, num_relations, train_loader = initialize(file_path, batch_size)
    
    
    # Initialize the model
    models = {
        "TransEUncertainty": TransEUncertainty(num_entities, num_relations, embedding_dim),
        "DistMultUncertainty": DistMultUncertainty(num_entities, num_relations, embedding_dim),
        "ComplExUncertainty": ComplExUncertainty(num_entities, num_relations, embedding_dim),
        "RotatEUncertainty": RotatEUncertainty(num_entities, num_relations, embedding_dim)
    }
    
    optimizers = {name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}
    
    training_loop(models, train_loader, optimizers, "loss", dataset, num_epochs, num_entities, embedding_dim, batch_size, margin, file_path, result_file)
        
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
    
    dataset, num_entities, num_relations, train_loader = initialize(file_path, batch_size)
    
    # Initialize the model
    models = {
        "TransEUncertainty": TransEUncertainty(num_entities, num_relations, embedding_dim),
        "DistMultUncertainty": DistMultUncertainty(num_entities, num_relations, embedding_dim),
        "ComplExUncertainty": ComplExUncertainty(num_entities, num_relations, embedding_dim),
    }
    optimizers = {name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}
        
    training_loop(models, train_loader, optimizers, "objective", dataset, num_epochs, num_entities, embedding_dim, batch_size, margin, file_path, result_file)
        
    train_and_evaluate_normal_models(dataset_models, embedding_dim, batch_size, num_epochs, margin, result_file=result_file)
    
def train_transE_with_different_losses(file_path, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset, num_entities, num_relations, train_loader = initialize(file_path, batch_size)
    
    # Initialize the model
    model = TransEUncertainty(num_entities, num_relations, embedding_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
                neg_triples = negative_sampling_creator.negative_sampling(list(zip(heads.cpu(), relations.cpu(), tails.cpu(), confidences.cpu())), num_entities)
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
        mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5 = evaluator.evaluate(model, dataset)

        # Print results
        print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_k}")

        # Write to CSV
        csvEditor.write_results_to_csv(
            result_file, name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k,
            file_path, total_loss, num_epochs, embedding_dim, batch_size, margin
        )
        
def train_distmult_with_different_losses(file_path, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset, num_entities, num_relations, train_loader = initialize(file_path, batch_size)
    
    # Initialize the model
    model = DistMultUncertainty(num_entities, num_relations, embedding_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
                neg_triples = negative_sampling_creator.negative_sampling(list(zip(heads.cpu(), relations.cpu(), tails.cpu(), confidences.cpu())), num_entities)
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
        mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5 = evaluator.evaluate(model, dataset)

        # Print results
        print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_k}")

        # Write to CSV
        csvEditor.write_results_to_csv(
            result_file, name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k,
            file_path, total_loss, num_epochs, embedding_dim, batch_size, margin
        )
        
def train_complex_with_different_losses(file_path, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset, num_entities, num_relations, train_loader = initialize(file_path, batch_size)
    
    # Initialize the model
    model = ComplExUncertainty(num_entities, num_relations, embedding_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
                neg_triples = negative_sampling_creator.negative_sampling(list(zip(heads.cpu(), relations.cpu(), tails.cpu(), confidences.cpu())), num_entities)
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
        mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5 = evaluator.evaluate_complex(model, dataset)

        # Print results
        print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_k}")

        # Write to CSV
        csvEditor.write_results_to_csv(
            result_file, name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k,
            file_path, total_loss, num_epochs, embedding_dim, batch_size, margin
        )