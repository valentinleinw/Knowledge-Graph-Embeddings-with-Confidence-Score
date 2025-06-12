# Knowledge-Graph-Embeddings-with-Confidence-Score

## Introduction

This project focuses on training Knowledge Graph Embedding models—specifically TransE, DistMult, and ComplEx—by incorporating confidence scores assigned to each triple in the dataset. It leverages existing [PyKEEN datasets](https://pykeen.readthedocs.io/en/stable/reference/datasets.html), where each triple is assigned a confidence score using various approaches. The models are then trained with different loss functions and evaluated using metrics such as Hits@k and Mean Reciprocal Rank (MRR).