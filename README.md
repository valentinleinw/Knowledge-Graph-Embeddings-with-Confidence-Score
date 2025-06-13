# Knowledge-Graph-Embeddings-with-Confidence-Score

## Introduction

This project focuses on training Knowledge Graph Embedding models—specifically TransE, DistMult, and ComplEx—by incorporating confidence scores assigned to each triple in the dataset. It leverages existing [PyKEEN datasets](https://pykeen.readthedocs.io/en/stable/reference/datasets.html), where each triple is assigned a confidence score using various approaches. The models are then trained with different loss functions and evaluated using metrics such as Hits@k and Mean Reciprocal Rank (MRR).

## How to run the code

### Creation of datasets

Even though some datasets were already created and are located in the 'datasets' folder, there is the option of creating new datasets if needed. All the necessary functions can be called in the main.py file. First, there are two functions from the paperApproach.py file, that are based on the approach from the UKGE model (see [here](https://arxiv.org/pdf/1811.10667)). For the execution of these two functions, only the required datasets needs to be specified.

```python
if __name__ == "__main__":

    paperApproach.logistic_function(ds.WN18RR())

    paperApproach.bounded_rectifier(ds.WN18RR())
```

The rest of the functions used for creating the datasets are located in the uncertaintyComputer.py file. Depedning on the required function, multiple parameters can be changed dynamically. For `add_confidence_score_based_on_appearances()` and `add_confidence_score_based_on_appearances_ranked()` only the datasets have to be specified, as already seen in the code snippet above. For `add_confidence_score_randomly()` the default range is defined as [0,1], but by specifying the start and/or the end, the range can be altered.

```python
if __name__ == "__main__":
    uncertaintyComputer.add_confidence_score_randomly(dataset8)

    uncertaintyComputer.add_confidence_score_randomly(dataset8, begin=0.5)

    uncertaintyComputer.add_confidence_score_randomly(dataset8, end=0.5)

    uncertaintyComputer.add_confidence_score_randomly(dataset8, begin=0.2, end=0.4)
```

### Training and evaluating the models