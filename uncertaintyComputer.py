import pykeen.datasets as ds
import pandas as pd
import os 
import numpy as np 
from pykeen.models import TransE, DistMult, ComplEx
from pykeen.pipeline import pipeline
from pykeen.models import ERModel
from pykeen.nn.representation import Embedding
from csvEditor import csvEditor


# for now using UMLS because it is pretty small
def addConfidenceScoreRandomly(begin=0, end=1):
    dataset = ds.UMLS()
    triples = np.concatenate([
        dataset.training.mapped_triples.numpy(),
        dataset.validation.mapped_triples.numpy(),
        dataset.testing.mapped_triples.numpy()
    ], axis=0)
    
    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
        
    # Generate random confidence scores and add to DataFrame
    df["confidence_score"] = np.random.rand(triples.shape[0]) * (end - begin) + begin

    save_to_csv(df, dataset, 0, "TransE", begin, end)
    
# I found the function in this paper: https://arxiv.org/pdf/1811.10667
def getEmbeddings(dataset, model_class):
    triples_factory = dataset.training
    
    model = model_class(
        triples_factory=triples_factory,
        embedding_dim=10 
    )
    
    result = pipeline(
        dataset=dataset,
        model=model,
        training_loop="slcwa",  
        optimizer="adam",  
        loss="margin",  
    )
    
    assert isinstance(model, ERModel)
    
    entity_embeddings = result.model.entity_representations[0]
    relation_embeddings = result.model.relation_representations[0]
    assert isinstance(entity_embeddings, Embedding)
    assert isinstance(relation_embeddings, Embedding)
    
    entity_embedding_tensor = entity_embeddings()
    relation_embedding_tensor = relation_embeddings()
    
    return entity_embedding_tensor.detach().numpy(), relation_embedding_tensor.detach().numpy()

def addConfidenceScoreBasedOnDataset(model_class, model_name):
    dataset = ds.UMLS()
    triples = np.concatenate([
        dataset.training.mapped_triples.numpy(),
        dataset.validation.mapped_triples.numpy(),
        dataset.testing.mapped_triples.numpy()
    ], axis=0)
    
    entity_embeddings, relation_embeddings = getEmbeddings(dataset, model_class)
    
    head_embeddings = entity_embeddings[triples[:, 0]] 
    relation_embeddings = relation_embeddings[triples[:, 1]]
    tail_embeddings = entity_embeddings[triples[:, 2]]
    
    combined_embeddings = np.multiply(head_embeddings, tail_embeddings)
    
    # Multiply the result with the relation embeddings (element-wise multiplication)
    final_confidence_scores = np.multiply(combined_embeddings, relation_embeddings).sum(axis=1)
    final_confidence_scores = np.abs(final_confidence_scores)
    final_confidence_scores = (final_confidence_scores - np.min(final_confidence_scores)) / (np.max(final_confidence_scores) - np.min(final_confidence_scores))
    
    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    df["confidence_score"] = final_confidence_scores
    
    csvEditor.save_to_csv(df, dataset, 1, model_name)

    
addConfidenceScoreRandomly(0.1, 0.2)
#addConfidenceScoreBasedOnDataset()
addConfidenceScoreBasedOnDataset(TransE, "TransE")
addConfidenceScoreBasedOnDataset(DistMult, "DistMult")
addConfidenceScoreBasedOnDataset(ComplEx, "ComplEx")
    
    