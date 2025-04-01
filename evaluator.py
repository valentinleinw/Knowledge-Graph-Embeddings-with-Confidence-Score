import torch
import numpy as np


def evaluate(model, test_loader, device='cpu', top_k=10):
    model.eval()  # Set model to evaluation mode

    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0

    # Get all entity and relation embeddings
    entity_embeddings = model.entity_embeddings.weight.to(device)  # (num_entities, embedding_dim)
    relation_embeddings = model.relation_embeddings.weight.to(device)  # (num_relations, embedding_dim)

    with torch.no_grad():  # Disable gradient calculation
        for batch in test_loader:
            heads, relations, tails, confidences = batch
            heads = heads.to(device)  # (batch_size,)
            relations = relations.to(device)  # (batch_size,)
            tails = tails.to(device)  # (batch_size,)
            confidences = confidences.to(device)  # (batch_size,)

            # Compute embeddings for the current batch
            head_embeddings = entity_embeddings[heads]  # (batch_size, embedding_dim)
            relation_embeddings_batch = relation_embeddings[relations]  # (batch_size, embedding_dim)
            tail_embeddings = entity_embeddings[tails]  # (batch_size, embedding_dim)

            # Compute scores for all entities (full set of entities)
            # Broadcasting head_embeddings and relation_embeddings_batch over all entities
            all_scores = torch.norm(
                head_embeddings.unsqueeze(1) + relation_embeddings_batch.unsqueeze(1) - entity_embeddings, p=1, dim=2
            )  # (batch_size, num_entities)

            # Get the rank of the correct tail entity for each instance in the batch
            for i in range(len(tails)):
                rank = (torch.argsort(all_scores[i], descending=False) == tails[i]).nonzero(as_tuple=True)[0].item() + 1  # 1-based rank
                ranks.append((rank, confidences[i]))

                # Check for Hits@1 and Hits@5
                hits_at_1 += (rank == 1)
                hits_at_5 += (rank <= 5)

    # Convert ranks to numpy for faster operations
    ranks = np.array(ranks, dtype=np.float32)
    mean_rank = np.mean(ranks[:, 0])
    mrr = np.mean(1.0 / ranks[:, 0])
    hits_at_k = np.mean(ranks[:, 0] <= top_k)

    # Normalize Hits@1 and Hits@5
    hits_at_1 /= len(ranks)
    hits_at_5 /= len(ranks)

    return mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5



def evaluate_complex(model, test_loader, device='cpu', top_k=10):
    model.eval()
    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0

    # Get all entity and relation embeddings
    entity_re_embeddings = model.entity_re_embeddings.weight.data.to(device)  # (num_entities, embedding_dim)
    entity_im_embeddings = model.entity_im_embeddings.weight.data.to(device)  # (num_entities, embedding_dim)
    relation_re_embeddings = model.relation_re_embeddings.weight.data.to(device)  # (num_relations, embedding_dim)
    relation_im_embeddings = model.relation_im_embeddings.weight.data.to(device)  # (num_relations, embedding_dim)

    with torch.no_grad():  # Disable gradient calculation
        for batch in test_loader:
            heads, relations, tails, confidences = batch
            heads = heads.to(device)  # (batch_size,)
            relations = relations.to(device)  # (batch_size,)
            tails = tails.to(device)  # (batch_size,)
            confidences = confidences.to(device)  # (batch_size,)

            # Extract the embeddings for the current batch
            head_real = entity_re_embeddings[heads]  # (batch_size, embedding_dim)
            head_imag = entity_im_embeddings[heads]  # (batch_size, embedding_dim)
            relation_real = relation_re_embeddings[relations]  # (batch_size, embedding_dim)
            relation_imag = relation_im_embeddings[relations]  # (batch_size, embedding_dim)

            # Compute all possible entity scores
            # Broadcasting head + relation embeddings over the full set of entity embeddings
            head_real_exp = head_real.unsqueeze(1)  # (batch_size, 1, embedding_dim)
            head_imag_exp = head_imag.unsqueeze(1)  # (batch_size, 1, embedding_dim)
            relation_real_exp = relation_real.unsqueeze(1)  # (batch_size, 1, embedding_dim)
            relation_imag_exp = relation_imag.unsqueeze(1)  # (batch_size, 1, embedding_dim)

            # Calculate the scores for all entities
            all_scores_re = torch.sum(
                (head_real_exp * relation_real_exp * entity_re_embeddings) +
                (head_imag_exp * relation_real_exp * entity_im_embeddings) +
                (head_real_exp * relation_imag_exp * entity_im_embeddings) -
                (head_imag_exp * relation_imag_exp * entity_re_embeddings),
                dim=2  # Sum over embedding_dim
            )  # (batch_size, num_entities)

            # For each batch, calculate the rank of the correct tail entity
            for i in range(len(tails)):
                rank = (torch.argsort(all_scores_re[i], descending=True) == tails[i]).nonzero(as_tuple=True)[0].item() + 1  # 1-based rank
                ranks.append((rank, confidences[i]))

                # Check for Hits@1 and Hits@5
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



"""
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
"""