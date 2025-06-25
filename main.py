import train
from datetime import datetime
import pykeen.datasets as ds
import time

if __name__ == "__main__":
    
    triples = [("datasets/paper_bounded_UMLS.csv", ds.UMLS(), f"results/results/paper_bounded_UMLS_results/evaluation_results")]
    
    for origin, dataset, result in triples:
        train.train_and_evaluate(origin, dataset, "loss", embedding_dim=200, batch_size=2048, num_epochs=1000, result_file=result + "_loss.csv")
        
        train.train_and_evaluate(origin, dataset, "objective", embedding_dim=200, batch_size=2048, num_epochs=1000, result_file=result + "_objective.csv")

        train.train_and_evaluate(origin, dataset, "contrastive", embedding_dim=200, batch_size=2048, num_epochs=1000, result_file=result + "_contrastive.csv")

        train.train_and_evaluate(origin, dataset, "divergence", embedding_dim=200, batch_size=2048, num_epochs=1000, result_file=result + "_divergence.csv")

        train.train_and_evaluate(origin, dataset, "gaussian", embedding_dim=200, batch_size=2048, num_epochs=1000, result_file=result + "_gaussian.csv")

        train.train_and_evaluate(origin, dataset, "softplus", embedding_dim=200, batch_size=2048, num_epochs=1000, result_file=result + "_softplus.csv")
        
        train.train_and_evaluate_neg_confidences_cosukg(origin, dataset, embedding_dim=200, batch_size=2048, num_epochs=1000, result_file=result + "_cosukg.csv")

        train.train_and_evaluate_neg_confidences_inverse(origin, dataset, embedding_dim=200, batch_size=2048, num_epochs=1000, result_file=result + "_inverse.csv")

        train.train_and_evaluate_neg_confidences_similarity(origin, dataset, embedding_dim=200, batch_size=2048, num_epochs=1000, result_file=result + "_similarity.csv")

    
    