import torch
import numpy as np

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