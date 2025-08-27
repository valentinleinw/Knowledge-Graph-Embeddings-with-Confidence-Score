import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx



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
    
def negative_sampling_cosukg(triples, num_entities, num_samples, x1, x2):
    neg_quad = []
    for head, relation, tail, confidence in triples:
        for _ in range(num_samples):
            if confidence > x1:
                # Generate (h, r, t, c') where c' is in (0, 1 - c]
                new_confidence = np.random.uniform(0, 1 - confidence)
                neg_quad.append((head, relation, tail, new_confidence))
            elif confidence < x2:
                # Generate (h, r, t, c') where c' is in [1 - c, 1)
                new_confidence = np.random.uniform(1 - confidence, 1)
                neg_quad.append((head, relation, tail, new_confidence))
            else:
                # Generate (h', r, t, c) or (h, r, t', c)
                if np.random.rand() > 0.5:
                    new_head = np.random.randint(num_entities)
                    neg_quad.append((new_head, relation, tail, confidence))
                else:
                    new_tail = np.random.randint(num_entities)
                    neg_quad.append((head, relation, new_tail, confidence))
    return neg_quad 

def negative_sampling_inverse(triples, num_entities, num_samples):
    neg_quad = []
    for head, relation, tail, confidence in triples:
        for _ in range(num_samples):
            # Generate a corrupted triple (h', r, t) or (h, r, t')
            if np.random.rand() > 0.5:
                new_head = np.random.randint(num_entities)
                new_tail = tail
            else:
                new_head = head
                new_tail = np.random.randint(num_entities)
            
            # Compute confidence as an inverse function of the original confidence
            new_confidence = 1 - confidence  
            neg_quad.append((new_head, relation, new_tail, new_confidence))

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

