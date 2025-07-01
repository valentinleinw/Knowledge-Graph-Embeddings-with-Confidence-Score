import pykeen.datasets as ds
import pandas as pd
import numpy as np 
from pykeen.models import TransE, DistMult, ComplEx
from pykeen.pipeline import pipeline
from pykeen.models import ERModel
from pykeen.nn.representation import Embedding
from csvEditor import csvEditor
from collections import defaultdict

# for now using UMLS because it is pretty small
def add_confidence_score_randomly(dataset, begin=0, end=1):
    triples = np.concatenate([
        dataset.training.mapped_triples.numpy(),
        dataset.validation.mapped_triples.numpy(),
        dataset.testing.mapped_triples.numpy()
    ], axis=0)
    
    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
        
    # Generate random confidence scores and add to DataFrame
    df["confidence_score"] = np.random.rand(triples.shape[0]) * (end - begin) + begin
    
    range = "[" + str(begin) + ";" + str(end) + "]"
    
    print(range)

    csvEditor.save_to_csv(df, dataset, "random", range=range)
    
def add_confidence_score_based_on_appearances(dataset):
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
            
    csvEditor.save_to_csv(df, dataset, "appearances")
    
# extreme uniform distribution of confidence scores, not good for real life examples
def add_confidence_score_based_on_appearances_ranked(dataset):
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
        
    df['confidence_score'] = df['raw_confidence'].rank(method='max', pct=True)
    
    df.drop(columns=['raw_confidence'], inplace=True)
    
    csvEditor.save_to_csv(df, dataset, "ranked_appearances")

# I found the function in this paper: https://arxiv.org/pdf/1811.10667
def get_embeddings(dataset, model_class, num_epochs, batch_size, embedding_dim):
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
        training_kwargs={
            'num_epochs': num_epochs,
            'batch_size': batch_size,
        },
        model_kwargs={
            'embedding_dim': embedding_dim,
        }
    )
    
    assert isinstance(model, ERModel)
    
    entity_embeddings = result.model.entity_representations[0]
    relation_embeddings = result.model.relation_representations[0]
    assert isinstance(entity_embeddings, Embedding)
    assert isinstance(relation_embeddings, Embedding)
    
    entity_embedding_tensor = entity_embeddings()
    relation_embedding_tensor = relation_embeddings()
    
    return entity_embedding_tensor.detach().numpy(), relation_embedding_tensor.detach().numpy()

def add_confidence_score_based_on_model(dataset, model_class, model_name, num_epochs, batch_size, embedding_dim):
    triples = np.concatenate([
        dataset.training.mapped_triples.numpy(),
        dataset.validation.mapped_triples.numpy(),
        dataset.testing.mapped_triples.numpy()
    ], axis=0)
    
    entity_embeddings, relation_embeddings = get_embeddings(dataset, model_class, num_epochs, batch_size, embedding_dim)
    
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
    
    csvEditor.save_to_csv(df, dataset, "model", model_name)

def add_confidence_score_based_on_dataset_average(dataset, num_epochs, batch_size, embedding_dim):    
    df_transE = compute_confidence_score(TransE, dataset, num_epochs, batch_size, embedding_dim)
    df_distMult = compute_confidence_score(DistMult, dataset, num_epochs, batch_size, embedding_dim)
    df_complEx = compute_confidence_score(ComplEx, dataset, num_epochs, batch_size, embedding_dim)
        
    df_average = compute_avg_confidence_score(df_transE, df_distMult, df_complEx)
    
    csvEditor.save_to_csv(df_average, dataset, "average")
    
def compute_avg_confidence_score(df_transE, df_distMult, df_complEx):
        df_average = df_transE.merge(df_distMult, on=['head', 'relation', 'tail'], how='inner', suffixes=('_transE', '_distMult')) \
               .merge(df_complEx, on=['head', 'relation', 'tail'], how='inner')

        df_average.rename(columns={'confidence_score': 'confidence_score_complEx'}, inplace=True)

        df_average['confidence_score'] = df_average[['confidence_score_transE', 'confidence_score_distMult', 'confidence_score_complEx']].mean(axis=1)
        
        df_average.drop(columns=['confidence_score_transE', 'confidence_score_distMult', 'confidence_score_complEx'], inplace=True)
        
        return df_average
    
def compute_confidence_score(model, dataset, num_epochs, batch_size, embedding_dim):
    triples = np.concatenate([
        dataset.training.mapped_triples.numpy(),
        dataset.validation.mapped_triples.numpy(),
        dataset.testing.mapped_triples.numpy()
    ], axis=0)
    
    entity_embeddings, relation_embeddings = get_embeddings(dataset, model, num_epochs, batch_size, embedding_dim)
    
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
   
def add_confidence_score_based_on_dataset_agreement(dataset, num_epochs, batch_size, embedding_dim):
    dataset
    
    threshold = 0.5
    
    df_transE = compute_confidence_score(TransE, dataset, num_epochs, batch_size, embedding_dim)
    df_distMult = compute_confidence_score(DistMult, dataset, num_epochs, batch_size, embedding_dim)
    df_complEx = compute_confidence_score(ComplEx, dataset, num_epochs, batch_size, embedding_dim)
        
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
    
    csvEditor.save_to_csv(df_all, dataset, "agree")
   
# might redo some logical rules, e.g. similarity rule because leads to many same confidence scores
def add_confidence_score_logical_rules(dataset, num_epochs, batch_size, embedding_dim):
        
    df_transE = compute_confidence_score(TransE, dataset, num_epochs, batch_size, embedding_dim)
    df_distMult = compute_confidence_score(DistMult, dataset, num_epochs, batch_size, embedding_dim)
    df_complEx = compute_confidence_score(ComplEx, dataset, num_epochs, batch_size, embedding_dim)
        
    df = compute_avg_confidence_score(df_transE, df_distMult, df_complEx)
    
    triples = {}
    for _, row in df.iterrows():
        triples[(row["head"], row["tail"])] = row["confidence_score"]
    
    def apply_transitivity_rule(df, triples):
        for index, row in df.iterrows():
            head, tail = row["head"], row["tail"]
            if (head, tail) in triples:  # Check if direct connection exists
                direct_conf = triples[(head, tail)]
                
                # Look for transitive relations
                for middle_entity in df['head']:
                    if (tail, middle_entity) in triples:  # If there's a second hop
                        transitive_conf = triples.get((tail, middle_entity), 0)
                        new_conf = min(direct_conf, transitive_conf)
                        # Boost the confidence of the third entity connection
                        triples[(head, middle_entity)] = max(triples.get((head, middle_entity), 0), new_conf)

        # Update the dataframe with new confidence scores
        for (h, t), conf in triples.items():
            df.loc[(df["head"] == h) & (df["tail"] == t), "confidence_score"] = conf
        return df
    
    def apply_similarity_rule(df):
        head_dict = defaultdict(list)

        # Populate the dictionary with head -> (tail, confidence)
        for _, row in df.iterrows():
            head_dict[row["head"]].append((row["tail"], row["confidence_score"]))
        
        # Boost confidence for similar head entities
        for head, connections in head_dict.items():
            for i in range(len(connections)):
                for j in range(i + 1, len(connections)):
                    tail_i, conf_i = connections[i]
                    tail_j, conf_j = connections[j]
                    
                    # Apply rule: Boost the confidence if both share the same head
                    new_conf = max(conf_i, conf_j)
                    if new_conf < 1:
                        df.loc[(df["head"] == head) & (df["tail"] == tail_j), "confidence_score"] = new_conf
                        df.loc[(df["head"] == head) & (df["tail"] == tail_i), "confidence_score"] = new_conf
        return df
    
    def apply_frequency_rule(df):
        entity_freq = pd.concat([df['head'], df['tail']]).value_counts()

        # Propagate confidence based on frequency
        for _, row in df.iterrows():
            head_freq = entity_freq.get(row["head"], 0)
            tail_freq = entity_freq.get(row["tail"], 0)
            
            # If the entities appear frequently, boost the confidence score
            if head_freq > 1 and tail_freq > 1:
                df.loc[(df["head"] == row["head"]) & (df["tail"] == row["tail"]), "confidence_score"] *= 1.2  # Boost by 20%
                df["confidence_score"] = df["confidence_score"].clip(upper=1)  # Ensure confidence doesn't exceed 1
        return df
    
    def apply_symmetry_rule(df, triples):
        # Iterate over each row in the dataset
        for index, row in df.iterrows():
            head, tail = row["head"], row["tail"]
            if (head, tail) in triples:
                direct_conf = triples[(head, tail)]
                
                # Check if the inverse (tail -> head) exists in the triples dictionary
                if (tail, head) in triples:
                    inverse_conf = triples[(tail, head)]
                    
                    # If the inverse exists and has a lower confidence, boost it to match the direct relation
                    if inverse_conf < direct_conf:
                        triples[(tail, head)] = direct_conf  # Set the same confidence for the inverse
                    else:
                        triples[(head, tail)] = inverse_conf

        
        # Update the DataFrame with new confidence scores for inverse relations
        for (h, t), conf in triples.items():
            df.loc[(df["head"] == h) & (df["tail"] == t), "confidence_score"] = conf
            
        return df
    
    def apply_all_rules(df, triples):
        # Apply all logical rules including symmetry
        df = apply_transitivity_rule(df, triples)  # Apply transitivity rule
        df = apply_similarity_rule(df)  # Apply similarity rule
        df = apply_frequency_rule(df)  # Apply frequency rule
        df = apply_symmetry_rule(df, triples)  # Apply symmetry rule

        # Normalize confidence scores after applying all rules
        df['confidence_score'] = (df['confidence_score'] - df['confidence_score'].min()) / (df['confidence_score'].max() - df['confidence_score'].min())
        
        return df
    
    df = apply_all_rules(df, triples)
    
    csvEditor.save_to_csv(df, dataset, "logical")

def add_confidence_score_logical_rules_with_distmult(dataset, num_epochs, batch_size, embedding_dim):
        
    df = compute_confidence_score(DistMult, dataset, num_epochs, batch_size, embedding_dim)
    
    triples = {}
    for _, row in df.iterrows():
        triples[(row["head"], row["tail"])] = row["confidence_score"]
    
    def apply_transitivity_rule(df, triples):
        for index, row in df.iterrows():
            head, tail = row["head"], row["tail"]
            if (head, tail) in triples:  # Check if direct connection exists
                direct_conf = triples[(head, tail)]
                
                # Look for transitive relations
                for middle_entity in df['head']:
                    if (tail, middle_entity) in triples:  # If there's a second hop
                        transitive_conf = triples.get((tail, middle_entity), 0)
                        new_conf = min(direct_conf, transitive_conf)
                        # Boost the confidence of the third entity connection
                        triples[(head, middle_entity)] = max(triples.get((head, middle_entity), 0), new_conf)

        # Update the dataframe with new confidence scores
        for (h, t), conf in triples.items():
            df.loc[(df["head"] == h) & (df["tail"] == t), "confidence_score"] = conf
        return df
    
    def apply_similarity_rule(df):
        head_dict = defaultdict(list)

        # Populate the dictionary with head -> (tail, confidence)
        for _, row in df.iterrows():
            head_dict[row["head"]].append((row["tail"], row["confidence_score"]))
        
        # Boost confidence for similar head entities
        for head, connections in head_dict.items():
            for i in range(len(connections)):
                for j in range(i + 1, len(connections)):
                    tail_i, conf_i = connections[i]
                    tail_j, conf_j = connections[j]
                    
                    # Apply rule: Boost the confidence if both share the same head
                    new_conf = max(conf_i, conf_j)
                    if new_conf < 1:
                        df.loc[(df["head"] == head) & (df["tail"] == tail_j), "confidence_score"] = new_conf
                        df.loc[(df["head"] == head) & (df["tail"] == tail_i), "confidence_score"] = new_conf
        return df
    
    def apply_frequency_rule(df):
        entity_freq = pd.concat([df['head'], df['tail']]).value_counts()

        # Propagate confidence based on frequency
        for _, row in df.iterrows():
            head_freq = entity_freq.get(row["head"], 0)
            tail_freq = entity_freq.get(row["tail"], 0)
            
            # If the entities appear frequently, boost the confidence score
            if head_freq > 1 and tail_freq > 1:
                df.loc[(df["head"] == row["head"]) & (df["tail"] == row["tail"]), "confidence_score"] *= 1.2  # Boost by 20%
                df["confidence_score"] = df["confidence_score"].clip(upper=1)  # Ensure confidence doesn't exceed 1
        return df
    
    def apply_symmetry_rule(df, triples):
        # Iterate over each row in the dataset
        for index, row in df.iterrows():
            head, tail = row["head"], row["tail"]
            if (head, tail) in triples:
                direct_conf = triples[(head, tail)]
                
                # Check if the inverse (tail -> head) exists in the triples dictionary
                if (tail, head) in triples:
                    inverse_conf = triples[(tail, head)]
                    
                    # If the inverse exists and has a lower confidence, boost it to match the direct relation
                    if inverse_conf < direct_conf:
                        triples[(tail, head)] = direct_conf  # Set the same confidence for the inverse
                    else:
                        triples[(head, tail)] = inverse_conf

        
        # Update the DataFrame with new confidence scores for inverse relations
        for (h, t), conf in triples.items():
            df.loc[(df["head"] == h) & (df["tail"] == t), "confidence_score"] = conf
            
        return df
    
    def apply_all_rules(df, triples):
        # Apply all logical rules including symmetry
        df = apply_transitivity_rule(df, triples)  # Apply transitivity rule
        df = apply_similarity_rule(df)  # Apply similarity rule
        df = apply_frequency_rule(df)  # Apply frequency rule
        df = apply_symmetry_rule(df, triples)  # Apply symmetry rule

        # Normalize confidence scores after applying all rules
        df['confidence_score'] = (df['confidence_score'] - df['confidence_score'].min()) / (df['confidence_score'].max() - df['confidence_score'].min())
        
        return df
    
    df = apply_all_rules(df, triples)
    
    csvEditor.save_to_csv(df, dataset, "logical_with_distmult")
  

    
dataset = ds.UMLS()
dataset2 = ds.AristoV4()
#dataset3 = ds.CN3l()
dataset4 = ds.CoDExSmall()
dataset5 = ds.DBpedia50()
dataset6 = ds.Kinships()
dataset7 = ds.CoDExMedium()
dataset8 = ds.WN18RR()
dataset9 = ds.CoDExLarge()
dataset10 = ds.YAGO310()

add_confidence_score_randomly(dataset9)

add_confidence_score_randomly(dataset9, begin=0.5)

add_confidence_score_randomly(dataset9, end=0.5)

add_confidence_score_based_on_model(dataset9, TransE, "TransE", 200, 2048, 500)

add_confidence_score_based_on_model(dataset9, DistMult, "DistMult", 200, 2048, 500)

add_confidence_score_based_on_model(dataset9, ComplEx, "ComplEx", 200, 2048, 500)

add_confidence_score_based_on_dataset_average(dataset9, 200, 2048, 500)

add_confidence_score_based_on_dataset_agreement(dataset9, 200, 2048, 500)

add_confidence_score_based_on_appearances(dataset9)

add_confidence_score_based_on_appearances_ranked(dataset9)

add_confidence_score_logical_rules(dataset9, 200, 2048, 500)

add_confidence_score_logical_rules_with_distmult(dataset9, 200, 2048, 500)

add_confidence_score_randomly(dataset10)

add_confidence_score_randomly(dataset10, begin=0.5)

add_confidence_score_randomly(dataset10, end=0.5)

add_confidence_score_based_on_model(dataset10, TransE, "TransE", 200, 2048, 500)

add_confidence_score_based_on_model(dataset10, DistMult, "DistMult", 200, 2048, 500)

add_confidence_score_based_on_model(dataset10, ComplEx, "ComplEx", 200, 2048, 500)

add_confidence_score_based_on_dataset_average(dataset10, 200, 2048, 500)

add_confidence_score_based_on_dataset_agreement(dataset10, 200, 2048, 500)

add_confidence_score_based_on_appearances(dataset10)

add_confidence_score_based_on_appearances_ranked(dataset10)

add_confidence_score_logical_rules(dataset10, 200, 2048, 500)

add_confidence_score_logical_rules_with_distmult(dataset10, 200, 2048, 500)