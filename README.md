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

    uncertaintyComputer.add_confidence_score_randomly(ds.WN18RR())

    uncertaintyComputer.add_confidence_score_randomly(ds.WN18RR(), begin=0.5)

    uncertaintyComputer.add_confidence_score_randomly(ds.WN18RR(), end=0.5)

    uncertaintyComputer.add_confidence_score_randomly(ds.WN18RR(), begin=0.2, end=0.4)
```

For all the other functions, the parameters used by the model need to be specified. These parameters include the embedding dimensions, batch size and number of epochs used by the model for the computation of the embeddings. 

```python
if __name__ == "__main__":

    uncertaintyComputer.add_confidence_score_based_on_dataset_average(ds.WN18RR(), num_epochs=200, batch_size=2048, embedding_dim=500)
```

When using `add_confidence_score_based_on_model()` the model that should be used, as well as the model name has to be speicified(this is only for having a unique dataset name):

```python
if __name__ == "__main__":

    uncertaintyComputer.add_confidence_score_based_on_model(ds.WN18RR(), ComplEx, "ComplEx", num_epochs=200, batch_size=2048, embedding_dim=500)

    uncertaintyComputer.add_confidence_score_based_on_model(ds.WN18RR(), TransE, "TransE", num_epochs=200, batch_size=2048, embedding_dim=500)
```

### Training and evaluating the models