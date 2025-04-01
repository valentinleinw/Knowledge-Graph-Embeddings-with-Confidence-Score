import numpy as np

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