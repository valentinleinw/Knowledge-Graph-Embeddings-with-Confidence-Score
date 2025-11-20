import torch
import numpy as np


def evaluate(model, test_loader, device='cpu', top_k=10, entity_batch_size=5000):
    model.eval()

    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0

    # Get all entity and relation embeddings
    entity_embeddings = model.entity_embeddings.weight.to(device)
    relation_embeddings = model.relation_embeddings.weight.to(device)

    num_entities = entity_embeddings.size(0)

    with torch.no_grad():
        for batch in test_loader:
            heads, relations, tails, confidences = batch
            heads = heads.to(device, dtype=torch.long)
            relations = relations.to(device, dtype=torch.long)
            tails = tails.to(device, dtype=torch.long)
            confidences = confidences.to(device, dtype=torch.float32)

            # Compute embeddings for the current batch
            head_embeddings = entity_embeddings[heads]                
            relation_embeddings_batch = relation_embeddings[relations]  

            # Combine head + relation
            head_plus_rel = head_embeddings + relation_embeddings_batch  
            head_plus_rel = head_plus_rel.unsqueeze(1)                   

            # Collect scores in chunks
            all_scores = []
            for start in range(0, num_entities, entity_batch_size):
                end = start + entity_batch_size
                ent_emb_chunk = entity_embeddings[start:end]
                ent_emb_chunk = ent_emb_chunk.unsqueeze(0)    

                # Compute L1 distance
                scores = torch.norm(head_plus_rel - ent_emb_chunk, p=1, dim=2)
                all_scores.append(scores)

            # Concatenate scores across all entity chunks
            all_scores = torch.cat(all_scores, dim=1)  

            # Rank computation (without full sort)
            for i in range(len(tails)):
                tail_idx = tails[i].item()
                tail_score = all_scores[i, tail_idx]

                # Rank = number of entities with score <= true tail score
                rank = (all_scores[i] <= tail_score).sum().item()

                ranks.append((rank, confidences[i].item()))

                hits_at_1 += (rank == 1)
                hits_at_5 += (rank <= 5)

    # Convert ranks to numpy for metrics
    ranks = np.array(ranks, dtype=np.float32)
    mean_rank = np.mean(ranks[:, 0])
    mrr = np.mean(1.0 / ranks[:, 0])
    hits_at_k = np.mean(ranks[:, 0] <= top_k)

    # Normalize Hits@1 and Hits@5
    hits_at_1 /= len(ranks)
    hits_at_5 /= len(ranks)

    return mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5

"""
Can not use the same evaluation for ComplEx as TransE and DistMult 
because of the imaginary dimension
"""
def evaluate_complex(model, test_loader, device='cpu', top_k=10, entity_batch_size=5000):
    model.eval()
    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0

    # Get all entity and relation embeddings
    entity_re_embeddings = model.entity_re_embeddings.weight.data.to(device)
    entity_im_embeddings = model.entity_im_embeddings.weight.data.to(device)
    relation_re_embeddings = model.relation_re_embeddings.weight.data.to(device)
    relation_im_embeddings = model.relation_im_embeddings.weight.data.to(device)

    num_entities = entity_re_embeddings.size(0)

    with torch.no_grad():
        for batch in test_loader:
            heads, relations, tails, confidences = batch
            heads = heads.to(device)
            relations = relations.to(device)
            tails = tails.to(device)
            confidences = confidences.to(device)

            head_real = entity_re_embeddings[heads]
            head_imag = entity_im_embeddings[heads]
            relation_real = relation_re_embeddings[relations]
            relation_imag = relation_im_embeddings[relations]

            # Expand dimensions for broadcasting
            head_real_exp = head_real.unsqueeze(1)  
            head_imag_exp = head_imag.unsqueeze(1)
            relation_real_exp = relation_real.unsqueeze(1)
            relation_imag_exp = relation_imag.unsqueeze(1)

            # Collect scores across entity batches
            all_scores = []
            for start in range(0, num_entities, entity_batch_size):
                end = start + entity_batch_size
                ent_re = entity_re_embeddings[start:end]  
                ent_im = entity_im_embeddings[start:end]

                # Expand entities to match batch
                ent_re = ent_re.unsqueeze(0)
                ent_im = ent_im.unsqueeze(0)

                scores = torch.sum(
                    (head_real_exp * relation_real_exp * ent_re) +
                    (head_imag_exp * relation_real_exp * ent_im) +
                    (head_real_exp * relation_imag_exp * ent_im) -
                    (head_imag_exp * relation_imag_exp * ent_re),
                    dim=2
                ) 

                all_scores.append(scores)

            # Concatenate along entity dimension
            all_scores_re = torch.cat(all_scores, dim=1)

            # For each batch, calculate the rank of the correct tail
            for i in range(len(tails)):
                sorted_indices = torch.argsort(all_scores_re[i], descending=True)
                rank = (sorted_indices == tails[i]).nonzero(as_tuple=True)[0].item() + 1
                ranks.append((rank, confidences[i]))

                hits_at_1 += (rank == 1)
                hits_at_5 += (rank <= 5)

    # Compute Evaluation Metrics
    mean_rank = np.mean([rank for rank, _ in ranks])
    mrr = np.mean([1 / rank for rank, _ in ranks])
    hits_at_k = np.mean([1 if rank <= top_k else 0 for rank, _ in ranks])

    hits_at_1 /= len(ranks)
    hits_at_5 /= len(ranks)

    return mean_rank, mrr, hits_at_k, hits_at_1, hits_at_5

@torch.no_grad()
def evaluate_mae(model, data_loader, device):
    model.eval()
    mae_sum = 0.0
    n = 0
    for batch in data_loader:
        h, r, t, conf = batch
        h, r, t, conf = h.to(device), r.to(device), t.to(device), conf.to(device, dtype=torch.float32)
        pred = model(h, r, t)
        mae_sum += torch.sum(torch.abs(pred - conf)).item()
        n += conf.size(0)
    return mae_sum / n

