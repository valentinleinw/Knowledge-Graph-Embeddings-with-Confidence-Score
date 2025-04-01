import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import csvEditor

class TransEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin):
        super(TransEModel, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def score(self, h, r, t):
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        return -torch.norm(h_emb + r_emb - t_emb, p=1, dim=1)
    

def generate_negative_samples(h, r, t, num_entities, n, q_set):
    negatives = []
    for _ in range(n):
        if np.random.rand() < 0.5:
            h_prime = np.random.randint(0, num_entities)
            neg_quad = (h_prime, r, t)
        else:
            t_prime = np.random.randint(0, num_entities)
            neg_quad = (h, r, t_prime)
        
        if neg_quad not in q_set:
            negatives.append(neg_quad)
    return negatives


def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    triples = [(row['head'], row['relation'], row['tail'], row['confidence_score']) for _, row in df.iterrows()]
    return triples


def train_transe(train_set, num_entities, num_relations, embedding_dim, batch_size, n, margin, learning_rate, num_epochs):
    model = TransEModel(num_entities, num_relations, embedding_dim, margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MarginRankingLoss(margin=margin)
    
    for epoch in range(num_epochs):
        indices = np.random.choice(len(train_set), batch_size, replace=False)
        batch = [train_set[i] for i in indices]
        
        loss = 0
        for h, r, t, c in batch:
            neg_samples = generate_negative_samples(h, r, t, num_entities, n, train_set)
            # Ensure tensors are of type torch.long
            pos_score = model.score(torch.tensor([h], dtype=torch.long), torch.tensor([r], dtype=torch.long), torch.tensor([t], dtype=torch.long))
            neg_scores = torch.stack([model.score(torch.tensor([h_n], dtype=torch.long), torch.tensor([r_n], dtype=torch.long), torch.tensor([t_n], dtype=torch.long)) 
                                      for h_n, r_n, t_n in neg_samples])
            
            # Expand pos_score to match the shape of neg_scores
            pos_score_expanded = pos_score.expand_as(neg_scores).squeeze()  # Remove extra dimensions
            
            # Convert neg_scores and target to 1D for compatibility
            neg_scores = neg_scores.squeeze()  # Remove extra dimensions
            target = torch.tensor([-1.0] * len(neg_samples), dtype=torch.float)
            
            # Compute loss with confidence weighting
            loss += loss_fn(pos_score_expanded, neg_scores, target) * c  # Confidence as weight
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return model

def evaluate_transe(model, test_set, num_entities):
    ranks = []
    for h, r, t, _ in test_set:
        scores = []
        for entity in range(num_entities):
            # Ensure tensors are of type torch.long for embeddings
            score = model.score(torch.tensor([h], dtype=torch.long), torch.tensor([r], dtype=torch.long), torch.tensor([entity], dtype=torch.long))
            scores.append((entity, score.item()))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        rank = [i for i, (e, _) in enumerate(scores) if e == t][0] + 1
        ranks.append(rank)
    
    mrr = np.mean([1.0 / r for r in ranks])
    hits_1 = np.mean([1.0 if r <= 1 else 0.0 for r in ranks])
    hits_5 = np.mean([1.0 if r <= 5 else 0.0 for r in ranks])
    hits_10 = np.mean([1.0 if r <= 10 else 0.0 for r in ranks])
    
    print(f"MRR: {mrr:.4f}, Hits@1: {hits_1:.4f}, Hits@5: {hits_5:.4f}, Hits@10: {hits_10:.4f}")
    csvEditor.csvEditor.write_results_to_csv("results/test-results.csv", "TransE", "N/A", mrr, hits_1, hits_5, hits_10, "paper_bounded", "N/A", "5", "6", "16", "1.0")
    return mrr, hits_1, hits_5, hits_10


if __name__ == "__main__":
    csv_file = "datasets/paper_bounded_UMLS.csv"  # Change this to your dataset path
    dataset = load_dataset(csv_file)
    num_entities = len(set([h for h, _, t, _ in dataset] + [t for h, _, t, _ in dataset]))
    num_relations = len(set([r for _, r, _, _ in dataset]))
    
    train_set = dataset[:int(0.8 * len(dataset))]
    test_set = dataset[int(0.8 * len(dataset)):]  # 80-20 split
    
    model = train_transe(train_set, num_entities, num_relations, embedding_dim=6, batch_size=16, n=10, margin=1.0, learning_rate=0.01, num_epochs=10)
    evaluate_transe(model, test_set, num_entities)
