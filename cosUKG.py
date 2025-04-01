import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from csvEditor import csvEditor

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
    
class ComplExModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin):
        super(ComplExModel, self).__init__()
        self.entity_embeddings_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_embeddings_imag = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings_real = nn.Embedding(num_relations, embedding_dim)
        self.relation_embeddings_imag = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin

        nn.init.xavier_uniform_(self.entity_embeddings_real.weight)
        nn.init.xavier_uniform_(self.entity_embeddings_imag.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_real.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_imag.weight)

    def score(self, h, r, t):
        h_real = self.entity_embeddings_real(h)
        h_imag = self.entity_embeddings_imag(h)
        t_real = self.entity_embeddings_real(t)
        t_imag = self.entity_embeddings_imag(t)
        r_real = self.relation_embeddings_real(r)
        r_imag = self.relation_embeddings_imag(r)

        score = torch.sum(h_real * t_real + h_imag * t_imag + r_real * (h_real * t_real + h_imag * t_imag) - r_imag * (h_imag * t_real - h_real * t_imag), dim=1)
        return score

class DistMultModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin):
        super(DistMultModel, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(self, h, r, t):
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        return torch.sum(h_emb * r_emb * t_emb, dim=1)

class RotatEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin):
        super(RotatEModel, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(self, h, r, t):
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        return torch.sum(h_emb * r_emb - t_emb, dim=1)

def generate_negative_samples(h, r, t, num_entities, n, q_set, confidence_score, x1, x2):
    negatives = []
    for _ in range(n):
        if confidence_score > x1:
            c_prime = np.random.uniform(0, 1 - confidence_score)
            neg_quad = (h, r, t, c_prime)
        elif confidence_score < x2:
            c_prime = np.random.uniform(1 - confidence_score, 1)
            neg_quad = (h, r, t, c_prime)
        else:
            if np.random.rand() < 0.5:
                h_prime = np.random.randint(0, num_entities)
                neg_quad = (h_prime, r, t, confidence_score)
            else:
                t_prime = np.random.randint(0, num_entities)
                neg_quad = (h, r, t_prime, confidence_score)
        
        if neg_quad not in q_set:
            negatives.append(neg_quad)
    return negatives



def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    triples = [(row['head'], row['relation'], row['tail'], row['confidence_score']) for _, row in df.iterrows()]
    return triples


def train_link_prediction(model_class, train_set, num_entities, num_relations, embedding_dim, batch_size, n, margin, learning_rate, num_epochs, x1, x2):
    model = model_class(num_entities, num_relations, embedding_dim, margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MarginRankingLoss(margin=margin)
    
    for epoch in range(num_epochs):
        indices = np.random.choice(len(train_set), batch_size, replace=False)
        batch = [train_set[i] for i in indices]
        
        loss = 0
        for h, r, t, c in batch:
            neg_samples = generate_negative_samples(h, r, t, num_entities, n, train_set, c, x1, x2)
            pos_score = model.score(torch.tensor([h], dtype=torch.long), torch.tensor([r], dtype=torch.long), torch.tensor([t], dtype=torch.long))
            neg_scores = torch.stack([model.score(torch.tensor([h_n], dtype=torch.long), torch.tensor([r_n], dtype=torch.long), torch.tensor([t_n], dtype=torch.long)) 
                                      for h_n, r_n, t_n, _ in neg_samples])
            
            # Expand pos_score to match the shape of neg_scores
            pos_score_expanded = pos_score.expand_as(neg_scores).squeeze()
            neg_scores = neg_scores.squeeze()
            target = torch.tensor([-1.0] * len(neg_samples), dtype=torch.float)
            
            loss += loss_fn(pos_score_expanded, neg_scores, target) * c  # Confidence as weight
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return model

def evaluate_link_prediction(model, test_set, num_entities):
    ranks = []
    for h, r, t, _ in test_set:
        scores = []
        for entity in range(num_entities):
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
    
    # Assuming csvEditor has a method that can take in the results and save them to CSV
    csvEditor.write_results_to_csv("results/test-results.csv", model.__class__.__name__, "N/A", mrr, hits_1, hits_5, hits_10, "datasets/paper_bounded.csv", "", "", "", "", "")
    return mrr, hits_1, hits_5, hits_10



if __name__ == "__main__":
    # Load dataset
    csv_file = "datasets/paper_bounded_UMLS.csv"  # Change this to your dataset path
    dataset = load_dataset(csv_file)
    
    # Number of entities and relations
    num_entities = len(set([h for h, _, t, _ in dataset] + [t for h, _, t, _ in dataset]))
    num_relations = len(set([r for _, r, _, _ in dataset]))
    
    # Train-test split (80-20)
    train_set = dataset[:int(0.8 * len(dataset))]
    test_set = dataset[int(0.8 * len(dataset)):]  # 80-20 split
    
    # Set hyperparameters
    embedding_dim = 6
    batch_size = 16
    n = 10
    margin = 1.0
    learning_rate = 0.01
    num_epochs = 10
    x1, x2 = 0.3, 0.7  # Example threshold values for confidence score adjustment
    
    # List of models to train
    models = {
        "TransE": TransEModel,
        "ComplEx": ComplExModel,
        "DistMult": DistMultModel,
        "RotatE": RotatEModel
    }

    # Train and evaluate each model
    for model_name, model_class in models.items():
        print(f"Training {model_name} model...")
        
        # Train the model
        model = train_link_prediction(model_class, train_set, num_entities, num_relations, embedding_dim, batch_size, n, margin, learning_rate, num_epochs, x1, x2)
        
        # Evaluate the model
        print(f"Evaluating {model_name} model...")
        evaluate_link_prediction(model, test_set, num_entities)