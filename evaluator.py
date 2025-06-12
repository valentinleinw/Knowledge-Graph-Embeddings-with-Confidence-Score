import torch
import numpy as np


def evaluate(model, test_loader, device='cpu', top_k=10):
    model.eval() 

    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0

    # Get all entity and relation embeddings
    entity_embeddings = model.entity_embeddings.weight.to(device)
    relation_embeddings = model.relation_embeddings.weight.to(device)

    with torch.no_grad():  # Disable gradient calculation
        for batch in test_loader:
            heads, relations, tails, confidences = batch
            heads = heads.to(device)
            relations = relations.to(device)
            tails = tails.to(device)
            confidences = confidences.to(device)

            # Compute embeddings for the current batch
            head_embeddings = entity_embeddings[heads]  # (batch_size, embedding_dim)
            relation_embeddings_batch = relation_embeddings[relations]  # (batch_size, embedding_dim)

            # Compute scores for all entities (full set of entities)
            # Broadcasting head_embeddings and relation_embeddings_batch over all entities
            all_scores = torch.norm(
                head_embeddings.unsqueeze(1) + relation_embeddings_batch.unsqueeze(1) - entity_embeddings, p=1, dim=2
            ) 

            # Get the rank of the correct tail entity for each instance in the batch
            for i in range(len(tails)):
                rank = (torch.argsort(all_scores[i], descending=False) == tails[i]).nonzero(as_tuple=True)[0].item() + 1
                ranks.append((rank, confidences[i]))

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

"""
Can not use the same evaluation for ComplEx as TransE and DistMult 
because of the imaginary dimension
"""
def evaluate_complex(model, test_loader, device='cpu', top_k=10):
    model.eval()
    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0

    # Get all entity and relation embeddings
    entity_re_embeddings = model.entity_re_embeddings.weight.data.to(device)  
    entity_im_embeddings = model.entity_im_embeddings.weight.data.to(device)  
    relation_re_embeddings = model.relation_re_embeddings.weight.data.to(device)  
    relation_im_embeddings = model.relation_im_embeddings.weight.data.to(device)  

    with torch.no_grad():  # Disable gradient calculation
        for batch in test_loader:
            heads, relations, tails, confidences = batch
            heads = heads.to(device)  
            relations = relations.to(device)  
            tails = tails.to(device)  
            confidences = confidences.to(device)  

            # Extract the embeddings for the current batch
            head_real = entity_re_embeddings[heads]  
            head_imag = entity_im_embeddings[heads]  
            relation_real = relation_re_embeddings[relations]  
            relation_imag = relation_im_embeddings[relations] 

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
                dim=2 
            ) 

            # For each batch, calculate the rank of the correct tail entity
            for i in range(len(tails)):
                rank = (torch.argsort(all_scores_re[i], descending=True) == tails[i]).nonzero(as_tuple=True)[0].item() + 1  # 1-based rank
                ranks.append((rank, confidences[i]))

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
