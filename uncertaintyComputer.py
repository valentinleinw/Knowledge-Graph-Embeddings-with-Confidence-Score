import pykeen.datasets as ds
import pandas as pd
import os 
import numpy as np 

# for now using UMLS because it is pretty small
def addConfidenceScoreRandomly():
    dataset = ds.UMLS()
    triples = np.concatenate([
        dataset.training.mapped_triples.numpy(),
        dataset.validation.mapped_triples.numpy(),
        dataset.testing.mapped_triples.numpy()
    ], axis=0)
    
    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
        
    # Generate random confidence scores and add to DataFrame
    df["confidence_score"] = np.random.rand(triples.shape[0])

    # Get dataset name dynamically
    dataset_name = dataset.__class__.__name__

    folder_path = "datasets"
    file_path = os.path.join(folder_path, f"{dataset_name}_with_uncertainty.csv")

    os.makedirs(folder_path, exist_ok=True)

    df.to_csv(file_path, index=False)
    
if __name__ == "__main__":
    addConfidenceScoreRandomly()
    
    