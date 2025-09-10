import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch



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
    
def negative_sampling_cosukg(triples, num_entities, num_samples, x1, x2, device):
    heads, rels, tails, confs = zip(*triples)
    heads = torch.tensor(heads, dtype=torch.long, device=device)
    rels = torch.tensor(rels, dtype=torch.long, device=device)
    tails = torch.tensor(tails, dtype=torch.long, device=device)
    confs = torch.tensor(confs, dtype=torch.float, device=device)

    heads = heads.repeat_interleave(num_samples)
    rels = rels.repeat_interleave(num_samples)
    tails = tails.repeat_interleave(num_samples)
    confs = confs.repeat_interleave(num_samples)

    new_confs = confs.clone()

    mask1 = confs > x1
    if mask1.any():
        new_confs[mask1] = torch.rand(mask1.sum(), device=device) * (1 - confs[mask1])

    mask2 = confs < x2
    if mask2.any():
        new_confs[mask2] = (1 - confs[mask2]) + torch.rand(mask2.sum(), device=device) * confs[mask2]

    mask3 = ~(mask1 | mask2)
    if mask3.any():
        corrupt_heads = torch.rand(mask3.sum(), device=device) > 0.5
        idx_mask3 = mask3.nonzero(as_tuple=True)[0]

        idx_heads = idx_mask3[corrupt_heads]
        rand_heads = torch.randint(0, num_entities, (idx_heads.size(0),), device=device)
        heads[idx_heads] = rand_heads

        idx_tails = idx_mask3[~corrupt_heads]
        rand_tails = torch.randint(0, num_entities, (idx_tails.size(0),), device=device)
        tails[idx_tails] = rand_tails

    # Return separate tensors with correct dtypes
    neg_triples = torch.stack([heads, rels, tails], dim=1)  # LongTensor
    neg_confidences = new_confs                             # FloatTensor

    return neg_triples, neg_confidences


def negative_sampling_inverse(triples, num_entities, num_samples, x1, x2, device):

    # Unpack list of triples into tensors
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

    # Case 3: otherwise → corrupt head or tail (inverse sampling logic)
    mask3 = ~(mask1 | mask2)
    if mask3.any():
        corrupt_heads = torch.rand(mask3.sum(), device=device) > 0.5
        idx_mask3 = mask3.nonzero(as_tuple=True)[0]

        # Corrupt heads
        idx_heads = idx_mask3[corrupt_heads]
        rand_heads = torch.randint(0, num_entities, (idx_heads.size(0),), device=device)
        heads[idx_heads] = rand_heads

        # Corrupt tails
        idx_tails = idx_mask3[~corrupt_heads]
        rand_tails = torch.randint(0, num_entities, (idx_tails.size(0),), device=device)
        tails[idx_tails] = rand_tails

    # Return separate tensors with correct dtypes
    neg_triples = torch.stack([heads, rels, tails], dim=1)  # LongTensor
    neg_confidences = new_confs                             # FloatTensor

    return neg_triples, neg_confidences



def precompute_similar_entities(entity_embeddings, top_k=10):
    similarity_matrix = cosine_similarity(entity_embeddings)
    top_similar = np.argpartition(similarity_matrix, -top_k, axis=1)[:, -top_k:]
    # Sort only the top_k candidates
    row_idx = np.arange(similarity_matrix.shape[0])[:, None]
    top_scores = similarity_matrix[row_idx, top_similar]
    sorted_idx = np.argsort(top_scores, axis=1)
    top_similar = top_similar[row_idx, sorted_idx]
    top_scores = top_scores[row_idx, sorted_idx]
    return top_similar, top_scores


def prepare_sampling_data(top_similar, similarity_scores):
    probas = similarity_scores / similarity_scores.sum(axis=1, keepdims=True)
    return probas


def negative_sampling_similarity(triples, num_samples, top_similar, similarity_scores):
    # Precompute probability distributions
    probas = prepare_sampling_data(top_similar, similarity_scores)

    n_triples = len(triples)

    # Expand triples: repeat each triple num_samples times
    triples = np.repeat(np.array(triples), num_samples, axis=0)

    heads, rels, tails, confs = triples.T
    heads = heads.astype(int)
    rels = rels.astype(int)
    tails = tails.astype(int)
    confs = confs.astype(float)

    # Randomly decide whether to replace head (True) or tail (False)
    replace_head = np.random.rand(len(triples)) > 0.5

    # Sample new heads
    sampled_heads = np.array([
        np.random.choice(top_similar[h], p=probas[h]) for h in heads[replace_head]
    ]) if replace_head.any() else np.array([])

    # Sample new tails
    sampled_tails = np.array([
        np.random.choice(top_similar[t], p=probas[t]) for t in tails[~replace_head]
    ]) if (~replace_head).any() else np.array([])

    # Assign replacements
    new_heads = heads.copy()
    new_tails = tails.copy()
    new_heads[replace_head] = sampled_heads
    new_tails[~replace_head] = sampled_tails

    # Lookup similarity scores in one vectorized step
    sim_scores = np.zeros(len(triples))
    sim_scores[replace_head] = [
        similarity_scores[h, np.where(top_similar[h] == nh)[0][0]]
        for h, nh in zip(heads[replace_head], new_heads[replace_head])
    ]
    sim_scores[~replace_head] = [
        similarity_scores[t, np.where(top_similar[t] == nt)[0][0]]
        for t, nt in zip(tails[~replace_head], new_tails[~replace_head])
    ]

    # New confidences
    new_confs = confs * sim_scores

    # Build final neg_quad
    neg_quad = np.stack([new_heads, rels, new_tails, new_confs], axis=1)

    return neg_quad


