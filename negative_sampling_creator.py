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
    
    triples = triples.to(device)
    heads, rels, tails, confs = triples.T

    # Expand triples (repeat each num_samples times)
    heads = heads.repeat_interleave(num_samples)
    rels = rels.repeat_interleave(num_samples)
    tails = tails.repeat_interleave(num_samples)
    confs = confs.repeat_interleave(num_samples)

    # Case 1: confidence > x1 → new_conf ∈ (0, 1 - c]
    mask1 = confs > x1
    new_confs = confs.clone()
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

        # Corrupt heads
        rand_heads = torch.randint(0, num_entities, (corrupt_heads.sum(),), device=device)
        heads[mask3][corrupt_heads] = rand_heads

        # Corrupt tails
        rand_tails = torch.randint(0, num_entities, ((~corrupt_heads).sum(),), device=device)
        tails[mask3][~corrupt_heads] = rand_tails

    # Stack results back
    neg_quad = torch.stack([heads, rels, tails, new_confs], dim=1)
    return neg_quad



def negative_sampling_inverse(triples, num_entities, num_samples):
    triples = np.array(triples, dtype=np.float32) 
    
    # Expand triples num_samples times
    expanded = np.repeat(triples, num_samples, axis=0)
    heads, rels, tails, confs = expanded.T
    
    # Preallocate outputs
    new_heads = heads.astype(np.int32).copy()
    new_rels = rels.astype(np.int32).copy()
    new_tails = tails.astype(np.int32).copy()
    new_confs = (1 - confs)  # inverse confidence (vectorized)
    
    # Decide whether to corrupt head or tail
    corrupt_heads = np.random.rand(expanded.shape[0]) > 0.5
    
    # Replace heads
    new_heads[corrupt_heads] = np.random.randint(0, num_entities, corrupt_heads.sum())
    
    # Replace tails
    new_tails[~corrupt_heads] = np.random.randint(0, num_entities, (~corrupt_heads).sum())
    
    # Stack into output array
    neg_quad = np.stack([new_heads, new_rels, new_tails, new_confs], axis=1)
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

