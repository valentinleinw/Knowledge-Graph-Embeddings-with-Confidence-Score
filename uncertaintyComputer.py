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
def add_confidence_score_randomly(begin=0, end=1):
    dataset = ds.UMLS()
    triples = np.concatenate([
        dataset.training.mapped_triples.numpy(),
        dataset.validation.mapped_triples.numpy(),
        dataset.testing.mapped_triples.numpy()
    ], axis=0)
    
    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
        
    # Generate random confidence scores and add to DataFrame
    df["confidence_score"] = np.random.rand(triples.shape[0]) * (end - begin) + begin

    csvEditor.save_to_csv(df, dataset, 0, "TransE", begin, end)
    
def add_confidence_score_based_on_appearances():
    dataset = ds.UMLS()
    triples = np.concatenate([
        dataset.training.mapped_triples.numpy(),
        dataset.validation.mapped_triples.numpy(),
        dataset.testing.mapped_triples.numpy()
    ], axis=0)
    
    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    
    # Count occurrences of (h, r), (r, t), and (h, t)
    hr_counts = df.groupby(['head', 'relation']).size().to_dict()
    rt_counts = df.groupby(['relation', 'tail']).size().to_dict()
    ht_counts = df.groupby(['head', 'tail']).size().to_dict()  

    # Compute confidence scores
    def compute_confidence(h, r, t):
        N_hr = hr_counts.get((h, r), 1)  # Default to 1 to avoid division by zero
        N_rt = rt_counts.get((r, t), 1)
        N_ht = ht_counts.get((h, t), 1)
    
        return 1 / (N_hr + N_rt + N_ht)
    
    df['confidence_score'] = df.apply(lambda row: compute_confidence(row['head'], row['relation'], row['tail']), axis=1)

    # Normalize scores
    df['confidence_score'] = (df['confidence_score'] - df['confidence_score'].min()) / (df['confidence_score'].max() - df['confidence_score'].min())
            
    csvEditor.save_to_csv(df, dataset, 1)
    
# extreme uniform distribution of confidence scores, not good for real life examples
def add_confidence_score_based_on_appearances_ranked():
    dataset = ds.UMLS()
    triples = np.concatenate([
        dataset.training.mapped_triples.numpy(),
        dataset.validation.mapped_triples.numpy(),
        dataset.testing.mapped_triples.numpy()
    ], axis=0)
    
    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    
    head_counts = df['head'].value_counts().to_dict()
    tail_counts = df['tail'].value_counts().to_dict()
    relation_counts = df['relation'].value_counts().to_dict()
    
    def compute_confidence(h, r, t):
        log_h = np.log1p(head_counts.get(h, 1))
        log_t = np.log1p(tail_counts.get(t, 1))
        log_r = np.log1p(relation_counts.get(r, 1))
        
        return (log_h + log_t + log_r) / 3  # Averaged for smoothing

    df['raw_confidence'] = df.apply(lambda row: compute_confidence(row['head'], row['relation'], row['tail']), axis=1)
        
    df['confidence'] = df['raw_confidence'].rank(method='max', pct=True)
    
    df.drop(columns=['raw_confidence'], inplace=True)
    
    csvEditor.save_to_csv(df, dataset, 1)
    

# I found the function in this paper: https://arxiv.org/pdf/1811.10667
def get_embeddings(dataset, model_class):
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

def add_confidence_score_based_on_dataset(model_class, model_name):
    dataset = ds.UMLS()
    triples = np.concatenate([
        dataset.training.mapped_triples.numpy(),
        dataset.validation.mapped_triples.numpy(),
        dataset.testing.mapped_triples.numpy()
    ], axis=0)
    
    entity_embeddings, relation_embeddings = get_embeddings(dataset, model_class)
    
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

def add_confidence_score_based_on_dataset_average():
    dataset = ds.UMLS()
    
    df_transE = compute_confidence_score(TransE, dataset)
    df_distMult = compute_confidence_score(DistMult, dataset)
    df_complEx = compute_confidence_score(ComplEx, dataset)
        
    df_average = df_transE.merge(df_distMult, on=['head', 'relation', 'tail'], how='inner', suffixes=('_transE', '_distMult')) \
               .merge(df_complEx, on=['head', 'relation', 'tail'], how='inner')

    df_average.rename(columns={'confidence_score': 'confidence_score_complEx'}, inplace=True)

    df_average['confidence_score_avg'] = df_average[['confidence_score_transE', 'confidence_score_distMult', 'confidence_score_complEx']].mean(axis=1)
    
    df_average.drop(columns=['confidence_score_transE', 'confidence_score_distMult', 'confidence_score_complEx'], inplace=True)
    
    csvEditor.save_to_csv(df_average, dataset, 1, "average")
    
def compute_confidence_score(model, dataset):
    triples = np.concatenate([
        dataset.training.mapped_triples.numpy(),
        dataset.validation.mapped_triples.numpy(),
        dataset.testing.mapped_triples.numpy()
    ], axis=0)
    
    entity_embeddings, relation_embeddings = get_embeddings(dataset, model)
    
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
    
    return df
   
def add_confidence_score_based_on_dataset_agreement():
    dataset = ds.UMLS()
    
    threshold = 0.5
    
    df_transE = compute_confidence_score(TransE, dataset)
    df_distMult = compute_confidence_score(DistMult, dataset)
    df_complEx = compute_confidence_score(ComplEx, dataset)
        
    df_all = df_transE.merge(df_distMult, on=['head', 'relation', 'tail'], how='inner', suffixes=('_transE', '_distMult')) \
               .merge(df_complEx, on=['head', 'relation', 'tail'], how='inner')
               
    df_all.rename(columns={'confidence_score': 'confidence_score_complEx'}, inplace=True)
    
    df_all['confidence_score'] = np.nan
    
    df_all['confidence_score'] = df_all.apply(lambda row: max(row['confidence_score_complEx'], row['confidence_score_distMult'], row['confidence_score_transE']) \
        if row['confidence_score_complEx'] >= threshold and row['confidence_score_distMult'] >= threshold \
            or row['confidence_score_complEx'] >= threshold and row['confidence_score_transE'] >= threshold \
                or row['confidence_score_distMult'] >= threshold and row['confidence_score_transE'] >= threshold  
        else min(row['confidence_score_complEx'], row['confidence_score_distMult'], row['confidence_score_transE']), axis=1)
    
    df_all.drop(columns=['confidence_score_transE', 'confidence_score_distMult', 'confidence_score_complEx'], inplace=True)
    
    csvEditor.save_to_csv(df_all, dataset, 1, "agree")
    
"""    
addConfidenceScoreRandomly(0.1, 0.2)
addConfidenceScoreBasedOnDataset(TransE, "TransE")
addConfidenceScoreBasedOnDataset(DistMult, "DistMult")
addConfidenceScoreBasedOnDataset(ComplEx, "ComplEx")
"""    

    
# 1. compute the scores for all three models and take the average (done)
# 2. compute the scores for all models and if they all give high scores then use the average high score, else use a lower score (done)
# 3. find entities and relations that appear the most and give the confidence score based on the appearance (not really a good way if we want to have a completely new dataset)
# 4. use PageRank for confidence score
# 5. use logical rules ( -> for example first use the confidence score computed by the models and then modify the scores based on this rules)

add_confidence_score_based_on_appearances_ranked()