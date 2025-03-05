import pykeen.datasets as ds
import pandas as pd
import os 
import numpy as np 
from pykeen.models import TransE, DistMult, ComplEx
import torch
from pykeen.pipeline import pipeline
from pykeen.models import ERModel
from pykeen.nn.representation import Embedding


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
    
    save_to_csv(df, dataset, 1, model_name)

def save_to_csv(df, dataset, methodType: int, model="", begin=None, end=None):
    dataset_name = dataset.__class__.__name__  # Get dataset name dynamically
    
    methodName = ""
    range = ""
    
    if methodType == 0:
        methodName = "random"
        range = f"_from_{begin}_to_{end}"
    elif methodType == 1:
        methodName = "compute"

    # Define file path for saving
    folder_path = "datasets"
    file_path = os.path.join(folder_path, f"{dataset_name}_{methodName}{range}_{model}_with_confidence.csv")

    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Save DataFrame to CSV
    df.to_csv(file_path, index=False)

    print(f"Dataset with confidence scores saved successfully at: {file_path}")

    
if __name__ == "__main__":
    addConfidenceScoreRandomly(0.1, 0.2)
    #addConfidenceScoreBasedOnDataset()
    addConfidenceScoreBasedOnDataset(TransE, "TransE")
    addConfidenceScoreBasedOnDataset(DistMult, "DistMult")
    addConfidenceScoreBasedOnDataset(ComplEx, "ComplEx")
    
    