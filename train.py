import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import csv
import os


# Step 1: Read and preprocess the CSV file
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

# Step 2: TransE Model Definition
class TransEUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransEUncertainty, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
    
    def forward(self, h, r, t):
        return self.entity_embeddings(h) + self.relation_embeddings(r) - self.entity_embeddings(t)
    
    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        pos_loss = torch.sum(confidence_scores * torch.clamp(
            margin + torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1) -
            torch.norm(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), p=1, dim=1), min=0))
        return pos_loss
    
class DistMultUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(DistMultUncertainty, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
    
    def forward(self, h, r, t):
        head_embedding = self.entity_embeddings(h)
        relation_embedding = self.relation_embeddings(r)
        tail_embedding = self.entity_embeddings(t)
        return torch.sum(head_embedding * relation_embedding * tail_embedding, dim=1)

    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        pos_score = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        pos_loss = confidence_scores * torch.clamp(margin - pos_score + neg_score, min=0)
        return pos_loss.sum()
    
class ComplExUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(ComplExUncertainty, self).__init__()
        self.entity_re_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.entity_im_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_re_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.relation_im_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, h, r, t):
        head_real, head_imag = self.entity_re_embeddings(h), self.entity_im_embeddings(h)
        relation_real, relation_imag = self.relation_re_embeddings(r), self.relation_im_embeddings(r)
        tail_real, tail_imag = self.entity_re_embeddings(t), self.entity_im_embeddings(t)

        return torch.sum(
            head_real * relation_real * tail_real + head_imag * relation_real * tail_imag + head_real * relation_imag * tail_imag - head_imag * relation_imag * tail_real,
            dim=1
        )

    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        pos_score = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        pos_loss = confidence_scores * torch.clamp(margin - pos_score + neg_score, min=0)
        return pos_loss.sum()



# Step 3: Negative Sampling
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

# Step 4: Evaluation Metrics (Confidence-weighted)
def evaluate(model, dataset, top_k=10):
    model.eval()
    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0
    for head, relation, tail, confidence in dataset:
        # Compute the scores for all entities as the tail
        scores = []
        for t in range(len(dataset.entity_to_idx)):
            # Access embeddings using model.entity_embeddings and model.relation_embeddings
            head_embedding = model.entity_embeddings(torch.tensor([head], dtype=torch.long))
            relation_embedding = model.relation_embeddings(torch.tensor([relation], dtype=torch.long))
            tail_embedding = model.entity_embeddings(torch.tensor([t], dtype=torch.long))
            
            # Calculate the score based on L1 norm (translation operation)
            score = torch.norm(head_embedding + relation_embedding - tail_embedding, p=1).item()
            scores.append(score)
        
        # Rank the tail entity and compute the rank of the correct tail
        ranked_entities = np.argsort(scores)
        rank = np.where(ranked_entities == tail)[0][0] + 1
        ranks.append((rank, confidence))
        
        # Check for Hits@1 and Hits@5
        if rank <= 1:
            hits_at_1 += 1
        if rank <= 5:
            hits_at_5 += 1
    
    # Calculate Mean Rank, Mean Reciprocal Rank, and Hits@k
    mean_rank = np.mean([rank for rank, _ in ranks])
    mrr = np.mean([1 / rank for rank, _ in ranks])
    hits_at_k = np.mean([1 if rank <= top_k else 0 for rank, _ in ranks])
    weighted_mrr = np.sum([1 / rank * conf for rank, conf in ranks]) / np.sum([conf for _, conf in ranks])
    
    # Calculate Hits@1 and Hits@5
    hits_at_1 = hits_at_1 / len(ranks)
    hits_at_5 = hits_at_5 / len(ranks)
    
    return mean_rank, mrr, hits_at_k, weighted_mrr, hits_at_1, hits_at_5

# Evaluation for ComplEx model
def evaluate_complex(model, dataset, top_k=10):
    model.eval()
    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0

    # Extract the real and imaginary embeddings for entities and relations once
    entity_re_embeddings = model.entity_re_embeddings.weight.data  # Real part of entity embeddings
    entity_im_embeddings = model.entity_im_embeddings.weight.data  # Imaginary part of entity embeddings
    relation_re_embeddings = model.relation_re_embeddings.weight.data  # Real part of relation embeddings
    relation_im_embeddings = model.relation_im_embeddings.weight.data  # Imaginary part of relation embeddings

    # Loop over all triples in the dataset
    for head, relation, tail, confidence in dataset:
        # Get the embeddings for the head, relation, and tail (real and imaginary parts)
        head_real, head_imag = entity_re_embeddings[head], entity_im_embeddings[head]
        relation_real, relation_imag = relation_re_embeddings[relation], relation_im_embeddings[relation]
        tail_real, tail_imag = entity_re_embeddings[tail], entity_im_embeddings[tail]

        scores = []

        # Compute scores for all entities as the tail
        for t in range(len(dataset.entity_to_idx)):
            # Get the real and imaginary embeddings of the candidate tail entity
            tail_real_t = entity_re_embeddings[t]
            tail_imag_t = entity_im_embeddings[t]

            # Compute the ComplEx score using the interaction of real and imaginary parts
            score = torch.norm(
                head_real + relation_real - tail_real, p=2
            ) + torch.norm(
                head_imag + relation_real - tail_imag, p=2
            )

            scores.append(score.item())

        # Rank the tail entity and compute the rank of the correct tail
        ranked_entities = np.argsort(scores)
        rank = np.where(ranked_entities == tail)[0][0] + 1  # +1 to account for 1-based rank
        ranks.append((rank, confidence))

        # Check for Hits@1 and Hits@5
        if rank <= 1:
            hits_at_1 += 1
        if rank <= 5:
            hits_at_5 += 1

    # Calculate Mean Rank, MRR, Hits@k, and Weighted MRR
    mean_rank = np.mean([rank for rank, _ in ranks])
    mrr = np.mean([1 / rank for rank, _ in ranks])
    hits_at_k = np.mean([1 if rank <= top_k else 0 for rank, _ in ranks])
    weighted_mrr = np.sum([1 / rank * conf for rank, conf in ranks]) / np.sum([conf for _, conf in ranks])

    # Calculate Hits@1 and Hits@5
    hits_at_1 = hits_at_1 / len(ranks)
    hits_at_5 = hits_at_5 / len(ranks)

    return mean_rank, mrr, hits_at_k, weighted_mrr, hits_at_1, hits_at_5



# Step 5: Function to write the evaluation results to a CSV file
def write_results_to_csv(file_name, model_name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k, weighted_mrr):
    # Check if the file exists to write headers only once
    file_exists = os.path.exists(file_name)

    # Open the file in append mode ('a') to add new rows instead of overwriting
    with open(file_name, mode="a", newline="") as file:
        fieldnames = ["Model", "Mean Rank", "MRR", "Hits@1", "Hits@5", "Hits@10", "Weighted MRR"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header only if the file doesn't already exist
        if not file_exists:
            writer.writeheader()

        # Write the results for the given model
        writer.writerow({
            "Model": model_name,
            "Mean Rank": mean_rank,
            "MRR": mrr,
            "Hits@1": hits_at_1,
            "Hits@5": hits_at_5,
            "Hits@10": hits_at_k,
            "Weighted MRR": weighted_mrr
        })

# Step 6: Main Training and Evaluation Loop (with writing results to CSV)
def train_and_evaluate(file_path, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    # Load the dataset
    dataset = KnowledgeGraphDataset(file_path)
    num_entities = len(dataset.entities)
    num_relations = len(dataset.relations)
    
    # Initialize the model
    models = {
        "TransE": TransEUncertainty(num_entities, num_relations, embedding_dim),
        "DistMult": DistMultUncertainty(num_entities, num_relations, embedding_dim),
        "ComplEx": ComplExUncertainty(num_entities, num_relations, embedding_dim),
    }
    optimizers = {name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}
    
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training Loop
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
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
    
    # Evaluate the model
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        if name == "ComplEx":
            mean_rank, mrr, hits_at_k, weighted_mrr, hits_at_1, hits_at_5 = evaluate_complex(models[name], dataset)
        else:
            mean_rank, mrr, hits_at_k, weighted_mrr, hits_at_1, hits_at_5 = evaluate(models[name], dataset)

        # Print results
        print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_k}, Weighted MRR: {weighted_mrr}")
        
        write_results_to_csv(result_file, name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k, weighted_mrr)

# Call the function with your dataset file and desired result file path
current_datetime = datetime.now()

# Optionally, you can format it as a string if needed
date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

train_and_evaluate("datasets/UMLS_compute_with_confidence.csv", embedding_dim=50, batch_size=64, num_epochs=10, result_file=f"results/evaluation_results_{date}.csv")

