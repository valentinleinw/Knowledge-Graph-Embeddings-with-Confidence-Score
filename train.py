import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import TransEUncertainty, DistMultUncertainty, ComplExUncertainty
from csvEditor import csvEditor
from pykeen.datasets import UMLS
from pykeen.models import TransE, DistMult, ComplEx
from pykeen.evaluation import LCWAEvaluationLoop
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
import pykeen.datasets as ds



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
    weighted_mrr = np.sum([1 / rank * conf for rank, conf in ranks]) / np.sum([conf for _, conf in ranks])

    # Normalize Hits@1 and Hits@5
    hits_at_1 /= len(ranks)
    hits_at_5 /= len(ranks)

    return mean_rank, mrr, hits_at_k, weighted_mrr, hits_at_1, hits_at_5


# Step 6: Main Training and Evaluation Loop (with writing results to CSV)
def train_and_evaluate(file_path, embedding_dim=50, batch_size=64, num_epochs=10, margin=1.0, result_file='evaluation_results.csv'):
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
            mean_rank, mrr, hits_at_k, weighted_mrr, hits_at_1, hits_at_5 = evaluate_complex(models[name], dataset)
        else:
            mean_rank, mrr, hits_at_k, weighted_mrr, hits_at_1, hits_at_5 = evaluate(models[name], dataset)

        # Print results
        print(f"{name} Results - Mean Rank: {mean_rank}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_k}, Weighted MRR: {weighted_mrr}")
        
        csvEditor.write_results_to_csv(result_file, name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k, weighted_mrr, file_path, loss_model, num_epochs, embedding_dim, batch_size, margin)
        
    train_and_evaluate_normal_models(embedding_dim, batch_size, num_epochs, margin, result_file=result_file)
    
def train_and_evaluate_normal_models(embedding_dim, batch_size, num_epochs, margin, result_file='evaluation_results.csv'):
    dataset = ds.CoDExSmall()
    training = dataset.training
    validation = dataset.validation
    testing = dataset.testing
    assert validation is not None
    
    model = TransE(triples_factory=training, embedding_dim=embedding_dim)
    helper_for_normal_models(model, "TransE", num_epochs, batch_size, result_file, embedding_dim, training, validation, testing)
    
    model = DistMult(triples_factory=training, embedding_dim=embedding_dim)
    helper_for_normal_models(model, "DistMult", num_epochs, batch_size, result_file, embedding_dim, training, validation, testing)
    
    model = ComplEx(triples_factory=training, embedding_dim=embedding_dim)
    helper_for_normal_models(model, "ComplEx", num_epochs, batch_size, result_file, embedding_dim, training, validation, testing)

    
def helper_for_normal_models(model, name, num_epochs, batch_size, result_file, embedding_dim, training, validation, testing):

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
    
    
    csvEditor.write_results_to_csv(result_file, name, "N/A", mrr, hits_at_1, hits_at_5, hits_at_10, "N/A", "UMLS", losses_per_epoch[-1], num_epochs, embedding_dim, batch_size, "N/A")
    