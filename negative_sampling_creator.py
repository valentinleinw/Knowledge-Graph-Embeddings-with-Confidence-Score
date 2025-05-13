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

def weighted_choice(candidates, weights):
    probabilities = weights / np.sum(weights)
    return np.random.choice(candidates, p=probabilities)

def negative_sampling_similarity(triples, num_samples, top_similar, similarity_scores):
    neg_quad = []

    for head, relation, tail, confidence in triples:
        for _ in range(num_samples):
            if np.random.rand() > 0.5:
                candidates = top_similar[head]
                weights = similarity_scores[head]
                new_head = weighted_choice(candidates, weights)
                new_tail = tail
                similarity_score = similarity_scores[head][np.where(candidates == new_head)[0][0]]
            else:
                candidates = top_similar[tail]
                weights = similarity_scores[tail]
                new_tail = weighted_choice(candidates, weights)
                new_head = head
                similarity_score = similarity_scores[tail][np.where(candidates == new_tail)[0][0]]
            
            new_confidence = confidence * similarity_score
            neg_quad.append((new_head, relation, new_tail, new_confidence))

    return neg_quad

def negative_sampling_graph(triples, num_samples):

    neg_quad = []

    # Build the graph from triples
    graph = nx.Graph()
    for head, relation, tail, _ in triples:
        graph.add_edge(head, tail)

    valid_entities = list(graph.nodes)  # Only use entities actually in the graph

    for head, relation, tail, _ in triples:
        for _ in range(num_samples):
            if np.random.rand() > 0.5:
                new_head = np.random.choice(valid_entities)
                new_tail = tail
            else:
                new_head = head
                new_tail = np.random.choice(valid_entities)

            # Check if both nodes exist in the graph
            if new_head in graph and new_tail in graph:
                try:
                    distance = nx.shortest_path_length(graph, source=new_head, target=new_tail)
                    new_confidence = np.exp(-distance)
                except nx.NetworkXNoPath:
                    new_confidence = 0.0
            else:
                new_confidence = 0.0

            neg_quad.append((new_head, relation, new_tail, new_confidence))

    return neg_quad

