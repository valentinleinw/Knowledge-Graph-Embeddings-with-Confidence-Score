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
    neg_triples = []
    for head, relation, tail, confidence in triples:
        for _ in range(num_samples):
            if confidence > x1:
                # Generate (h, r, t, c') where c' is in (0, 1 - c]
                new_confidence = np.random.uniform(0, 1 - confidence)
                neg_triples.append((head, relation, tail, new_confidence))
            elif confidence < x2:
                # Generate (h, r, t, c') where c' is in [1 - c, 1)
                new_confidence = np.random.uniform(1 - confidence, 1)
                neg_triples.append((head, relation, tail, new_confidence))
            else:
                # Generate (h', r, t, c) or (h, r, t', c)
                if np.random.rand() > 0.5:
                    new_head = np.random.randint(num_entities)
                    neg_triples.append((new_head, relation, tail, confidence))
                else:
                    new_tail = np.random.randint(num_entities)
                    neg_triples.append((head, relation, new_tail, confidence))
    return neg_triples 

def negative_sampling_inverse(triples, num_entities, num_samples):
    neg_triples = []
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
            neg_triples.append((new_head, relation, new_tail, new_confidence))

    return neg_triples

def negative_sampling_similarity(triples, num_entities, num_samples, entity_embeddings):
    neg_triples = []
    
    for head, relation, tail, confidence in triples:
        for _ in range(num_samples):
            if np.random.rand() > 0.5:
                # Find a similar entity to replace the head
                similarities = cosine_similarity(entity_embeddings[head].reshape(1, -1), entity_embeddings)
                candidate_entities = np.argsort(similarities[0])[-10:]  # Top 10 similar entities
                new_head = np.random.choice(candidate_entities)
                new_tail = tail
            else:
                # Find a similar entity to replace the tail
                similarities = cosine_similarity(entity_embeddings[tail].reshape(1, -1), entity_embeddings)
                candidate_entities = np.argsort(similarities[0])[-10:]
                new_head = head
                new_tail = np.random.choice(candidate_entities)
            
            # Assign a confidence score based on similarity
            similarity_score = similarities[0][new_tail] if new_head == head else similarities[0][new_head]
            new_confidence = confidence * similarity_score  # Higher similarity → higher confidence

            neg_triples.append((new_head, relation, new_tail, new_confidence))
    
    return neg_triples

def negative_sampling_graph(triples, num_entities, num_samples, graph):
    neg_triples = []
    
    for head, relation, tail, confidence in triples:
        for _ in range(num_samples):
            if np.random.rand() > 0.5:
                new_head = np.random.randint(num_entities)
                new_tail = tail
            else:
                new_head = head
                new_tail = np.random.randint(num_entities)
            
            # Compute shortest path distance in the graph
            try:
                distance = nx.shortest_path_length(graph, source=new_head, target=new_tail)
            except nx.NetworkXNoPath:
                distance = np.inf  # No path means completely disconnected
            
            # Assign confidence based on distance (closer nodes → higher confidence)
            new_confidence = np.exp(-distance)  # Exponential decay function
            
            neg_triples.append((new_head, relation, new_tail, new_confidence))
    
    return neg_triples