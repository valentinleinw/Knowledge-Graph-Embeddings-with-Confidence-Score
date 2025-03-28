import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
import pandas as pd
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransEConfidence(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransEConfidence, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.scoring_layer = nn.Linear(embedding_dim, 1)  # Predicts confidence
        
    def forward(self, h, r, t):
        score = self.entity_embeddings(h) + self.relation_embeddings(r) - self.entity_embeddings(t)
        return score  # Now returns a tensor of shape (batch_size, embedding_dim)
    
    def predict_confidence(self, h, r, t):
        score = self.forward(h, r, t)
        confidence = torch.sigmoid(self.scoring_layer(score)).squeeze()
        return confidence
    
    def loss(self, triples, confidence_scores):
        predicted_confidence = self.predict_confidence(triples[:, 0], triples[:, 1], triples[:, 2])
        return nn.MSELoss()(predicted_confidence.view(-1, 1), confidence_scores.view(-1, 1))


def load_and_split_data(file_path, train_ratio=0.8, val_ratio=0.1):
    # Check first few rows of the raw dataset
    with open(file_path, 'r') as f:
        print(f.readline())  # Print the first line to check the separator and format

    # Read the dataset with tab-separated values
    df = pd.read_csv(file_path)

    # Ensure that the columns are correctly formatted
    df[['head', 'relation', 'tail']] = df[['head', 'relation', 'tail']].astype(str)

    # Map entities and relations to unique indices
    entity_to_id = {entity: idx for idx, entity in enumerate(set(df['head']).union(set(df['tail'])))}
    relation_to_id = {relation: idx for idx, relation in enumerate(set(df['relation']))}
    
    # Convert dataset to integer indices safely
    dataset = []
    for h, r, t, c in df.values:
        dataset.append((entity_to_id[h], relation_to_id[r], entity_to_id[t], c))

    dataset = torch.tensor(dataset, dtype=torch.float)

    # Ensure dataset has valid samples
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("Dataset has no samples after mapping to ids!")
    
    # Split data
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    
    return train_data, val_data, test_data, len(entity_to_id), len(relation_to_id)

def train_model(file_path, embedding_dim=50, batch_size=64, num_epochs=10, lr=0.001, result_file='results.csv'):
    train_data, val_data, test_data, num_entities, num_relations = load_and_split_data(file_path)
    model = TransEConfidence(num_entities, num_relations, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            heads, relations, tails, confidences = batch[:, 0].long(), batch[:, 1].long(), batch[:, 2].long(), batch[:, 3].float()
            
            optimizer.zero_grad()
            triples = torch.stack([heads, relations, tails], dim=1)
            loss = model.loss(triples, confidences)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
    
    print("\nEvaluating Model...")
    mse, mae, preds = evaluate_model(model, test_data)
    save_results_to_csv(model, embedding_dim, batch_size, num_epochs, lr, test_data, mse, mae, preds, result_file)

def evaluate_model(model, test_data):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actual_confidences = []
    
    with torch.no_grad():
        for h, r, t, actual_conf in test_data:
            h, r, t = torch.tensor(h, dtype=torch.long), torch.tensor(r, dtype=torch.long), torch.tensor(t, dtype=torch.long)
            predicted_conf = model.predict_confidence(h, r, t).item()
            
            # Store the predicted confidence and the actual confidence score
            predictions.append(predicted_conf)
            actual_confidences.append(actual_conf)
    
    # Convert lists to numpy arrays for easier calculation of metrics
    predictions = np.array(predictions)
    actual_confidences = np.array(actual_confidences)
    
    # Calculate Mean Squared Error (MSE) or Mean Absolute Error (MAE)
    mse = np.mean((predictions - actual_confidences) ** 2)  # MSE
    mae = np.mean(np.abs(predictions - actual_confidences))  # MAE
    
    return mse, mae, predictions

def save_results_to_csv(model, embedding_dim, batch_size, num_epochs, lr, test_data, mse, mae, predictions, result_file):
    with open(result_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write model details and parameters
        writer.writerow(["Model", "Embedding Dim", "Batch Size", "Epochs", "Learning Rate"])
        writer.writerow([model.__class__.__name__, embedding_dim, batch_size, num_epochs, lr])
        
        # Write the evaluation metrics (MSE and MAE)
        writer.writerow([])
        writer.writerow(["Mean Squared Error", mse])
        writer.writerow(["Mean Absolute Error", mae])
        
        # Write headers for the predictions
        writer.writerow([])
        writer.writerow(["Head", "Relation", "Tail", "Predicted Confidence", "Actual Confidence"])
        
        # Write the predictions and actual confidence values
        for (h, r, t, actual_conf), pred_conf in zip(test_data, predictions):
            writer.writerow([int(h.item()), int(r.item()), int(t.item()), pred_conf, actual_conf.item()])
    
    print(f"Results saved to {result_file}")


