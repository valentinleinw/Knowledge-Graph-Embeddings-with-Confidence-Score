# Knowledge-Graph-Embeddings-with-Confidence-Score

## Introduction

This project focuses on training Knowledge Graph Embedding models—specifically TransE, DistMult, and ComplEx—by incorporating confidence scores assigned to each triple in the dataset. It leverages existing [PyKEEN datasets](https://pykeen.readthedocs.io/en/stable/reference/datasets.html), where each triple is assigned a confidence score using various approaches. The models are then trained with different loss functions and evaluated using metrics such as Hits@k and Mean Reciprocal Rank (MRR).

## How to run the code

### Creation of datasets

Even though some datasets were already created and are located in the 'datasets' folder, there is the option of creating new datasets if needed. All the necessary functions can be called in the main.py file. First, there are two functions from the paperApproach.py file, that are based on the approach from the UKGE model (see [here](https://iopscience.iop.org/article/10.1088/1742-6596/1824/1/012002?)). For the execution of these two functions, only the required datasets needs to be specified.

```python
if __name__ == "__main__":

    paperApproach.logistic_function(ds.WN18RR())

    paperApproach.bounded_rectifier(ds.WN18RR())
```

### Training and evaluating the models