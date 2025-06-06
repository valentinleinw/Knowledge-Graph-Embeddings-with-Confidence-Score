import torch
from pykeen.pipeline import pipeline
import pykeen.datasets as ds
import torch.nn as nn
from csvEditor import csvEditor
import numpy as np


# Define the confidence score transformation using the sigmoid function
class ConfidenceTransformLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))  # Learnable scaling parameter
        self.b = nn.Parameter(torch.tensor(0.0))  # Learnable bias parameter

    def forward(self, plausibility):
        return 1 / (1 + torch.exp(-(self.w * plausibility + self.b)))
    
class ConfidenceTransformBounded(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))  # Learnable scaling parameter
        self.b = nn.Parameter(torch.tensor(0.0))  # Learnable bias parameter

    def forward(self, plausibility):
        return torch.clamp(self.w * plausibility + self.b, min=0, max=1)


def compute_embeddings(dataset):
    
    result = pipeline(
        model='DistMult',
        dataset=dataset,
        training_kwargs={'num_epochs': 200, 'batch_size': 2048},
        model_kwargs={'embedding_dim': 500,}
    )
    
    model = result.model
    
    triples = torch.cat([
        dataset.training.mapped_triples,
        dataset.validation.mapped_triples,
        dataset.testing.mapped_triples
    ])

    plausibility = model.score_hrt(triples)
    
    return plausibility, triples

def logistic_function(dataset):
    plausibility, triples = compute_embeddings(dataset)
    
    confidence_transform = ConfidenceTransformLogistic()

    # Apply sigmoid transformation to obtain confidence scores
    confidence_scores = confidence_transform(plausibility)
    
    csvEditor.save_to_csv_paper(dataset, confidence_scores, triples, "logistic")
    
def bounded_rectifier(dataset):
    plausibility, triples = compute_embeddings(dataset)
    
    confidence_transform = ConfidenceTransformBounded()

    # Apply sigmoid transformation to obtain confidence scores
    confidence_scores = confidence_transform(plausibility)
    
    csvEditor.save_to_csv_paper(dataset, confidence_scores, triples, "bounded")
    
logistic_function(ds.WN18RR())
bounded_rectifier(ds.WN18RR())
    
    
    
