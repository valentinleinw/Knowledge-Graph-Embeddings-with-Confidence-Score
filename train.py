import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


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
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
    
    def forward(self, h, r, t):
        return self.entity_embeddings(h) + self.relation_embeddings(r) - self.entity_embeddings(t)
    
    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        pos_loss = torch.sum(confidence_scores * torch.clamp(
            margin + torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1) -
            torch.norm(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), p=1, dim=1), min=0))
        return pos_loss

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


import csv

# Step 5: Function to write the evaluation results to a CSV file
def write_results_to_csv(file_name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k, weighted_mrr):
    # Create a dictionary with the evaluation metrics
    results = {
        'Mean Rank': mean_rank,
        'MRR': mrr,
        'Hits@1': hits_at_1,
        'Hits@5': hits_at_5,
        'Hits@10': hits_at_k,
        'Weighted MRR': weighted_mrr
    }
    
    # Write the results to a new CSV file
    with open(file_name, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        
        # Write the header
        writer.writeheader()
        
        # Write the results
        writer.writerow(results)

# Step 6: Main Training and Evaluation Loop (with writing results to CSV)
def train_and_evaluate(file_path, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
    # Load the dataset
    dataset = KnowledgeGraphDataset(file_path)
    num_entities = len(dataset.entities)
    num_relations = len(dataset.relations)
    
    # Initialize the model
    model = TransE(num_entities, num_relations, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Unpack the batch
            heads, relations, tails, confidences = batch
            
            # Convert to tensors
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
            
            # Compute loss
            optimizer.zero_grad()
            pos_triples = torch.stack([heads, relations, tails], dim=1)
            neg_triples = torch.stack([neg_heads, neg_relations, neg_tails], dim=1)
            loss = model.loss(pos_triples, neg_triples, confidences, margin)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
    
    # Evaluate the model
    mean_rank, mrr, hits_at_k, weighted_mrr, hits_at_1, hits_at_5 = evaluate(model, dataset)
    print(f"Evaluation Results: Mean Rank = {mean_rank}, MRR = {mrr}, Hits@1 = {hits_at_1}, Hits@5 = {hits_at_5}, Hits@10 = {hits_at_k}, Weighted MRR = {weighted_mrr}")
    
    # Write the results to a CSV file
    write_results_to_csv(result_file, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k, weighted_mrr)

# Call the function with your dataset file and desired result file path
current_datetime = datetime.now()

# Optionally, you can format it as a string if needed
date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

train_and_evaluate("datasets/UMLS_compute_with_confidence.csv", embedding_dim=50, batch_size=64, num_epochs=10, result_file=f"results/evaluation_results_{date}.csv")

