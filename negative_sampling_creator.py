import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



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
    
import torch

def negative_sampling_cosukg(triples, num_entities, num_samples, x1, x2, device):

    # Unpack list of triples into separate tensors
    heads, rels, tails, confs = zip(*triples)
    heads = torch.tensor(heads, dtype=torch.long, device=device)
    rels = torch.tensor(rels, dtype=torch.long, device=device)
    tails = torch.tensor(tails, dtype=torch.long, device=device)
    confs = torch.tensor(confs, dtype=torch.float, device=device)

    # Repeat each triple num_samples times
    heads = heads.repeat_interleave(num_samples)
    rels = rels.repeat_interleave(num_samples)
    tails = tails.repeat_interleave(num_samples)
    confs = confs.repeat_interleave(num_samples)

    # Initialize new confidences
    new_confs = confs.clone()

    # Case 1: confidence > x1 → new_conf ∈ (0, 1 - c]
    mask1 = confs > x1
    if mask1.any():
        new_confs[mask1] = torch.rand(mask1.sum(), device=device) * (1 - confs[mask1])

    # Case 2: confidence < x2 → new_conf ∈ [1 - c, 1)
    mask2 = confs < x2
    if mask2.any():
        new_confs[mask2] = (1 - confs[mask2]) + torch.rand(mask2.sum(), device=device) * confs[mask2]

    # Case 3: otherwise → corrupt head or tail
    mask3 = ~(mask1 | mask2)
    if mask3.any():
        corrupt_heads = torch.rand(mask3.sum(), device=device) > 0.5

        # Indices for corruption
        idx_mask3 = mask3.nonzero(as_tuple=True)[0]

        # Corrupt heads
        idx_heads = idx_mask3[corrupt_heads]
        rand_heads = torch.randint(0, num_entities, (idx_heads.size(0),), device=device)
        heads[idx_heads] = rand_heads

        # Corrupt tails
        idx_tails = idx_mask3[~corrupt_heads]
        rand_tails = torch.randint(0, num_entities, (idx_tails.size(0),), device=device)
        tails[idx_tails] = rand_tails

    # Stack results into shape (num_triples*num_samples, 4)
    neg_quad = torch.stack([heads, rels, tails, new_confs], dim=1)
    return neg_quad




import torch

def negative_sampling_inverse(triples, num_entities, num_samples, device):
    
    triples = triples.to(device)
    heads, rels, tails, confs = triples.T

    batch_size = triples.size(0)

    # Expand triples num_samples times
    heads = heads.repeat_interleave(num_samples)
    rels = rels.repeat_interleave(num_samples)
    tails = tails.repeat_interleave(num_samples)
    confs = confs.repeat_interleave(num_samples)

    # Inverse confidence
    new_confs = 1.0 - confs

    # Decide whether to corrupt head or tail
    corrupt_heads = torch.rand(batch_size * num_samples, device=device) > 0.5

    # Replace heads
    if corrupt_heads.any():
        rand_heads = torch.randint(0, num_entities, (corrupt_heads.sum(),), device=device)
        heads[corrupt_heads] = rand_heads

    # Replace tails
    if (~corrupt_heads).any():
        rand_tails = torch.randint(0, num_entities, ((~corrupt_heads).sum(),), device=device)
        tails[~corrupt_heads] = rand_tails

    # Stack back into quads
    neg_quad = torch.stack([heads, rels, tails, new_confs], dim=1)
    return neg_quad


def precompute_similar_entities(entity_embeddings, top_k=10):
    similarity_matrix = cosine_similarity(entity_embeddings)
    top_similar = np.argsort(similarity_matrix, axis=1)[:, -top_k:]
    similarity_scores = np.take_along_axis(similarity_matrix, top_similar, axis=1)
    return top_similar, similarity_scores

def prepare_sampling_data(top_similar, similarity_scores):

    n_entities, k = top_similar.shape

    probas = similarity_scores / similarity_scores.sum(axis=1, keepdims=True)

    # build index maps (list of dicts)
    index_maps = []
    for i in range(n_entities):
        index_maps.append({c: j for j, c in enumerate(top_similar[i])})

    return probas, index_maps


def weighted_choice(candidates, probabilities):
    return np.random.choice(candidates, p=probabilities)


def negative_sampling_similarity(triples, num_samples, top_similar, similarity_scores):
    neg_quad = []

    # Precompute probas + index maps
    probas, index_maps = prepare_sampling_data(top_similar, similarity_scores)

    for head, relation, tail, confidence in triples:
        for _ in range(num_samples):
            if np.random.rand() > 0.5:
                # replace head
                candidates = top_similar[head]
                new_head = weighted_choice(candidates, probas[head])
                new_tail = tail
                similarity_score = similarity_scores[head][index_maps[head][new_head]]
            else:
                # replace tail
                candidates = top_similar[tail]
                new_tail = weighted_choice(candidates, probas[tail])
                new_head = head
                similarity_score = similarity_scores[tail][index_maps[tail][new_tail]]

            new_confidence = confidence * similarity_score
            neg_quad.append((new_head, relation, new_tail, new_confidence))

    return neg_quad

