import train
from datetime import datetime
import pykeen.datasets as ds
import time

if __name__ == "__main__":
    
    triples = [
        ("datasets/paper_bounded_UMLS.csv", ds.UMLS(), f"results/results/paper_bounded_UMLS_results/evaluation_results"),
        ("datasets/paper_logistic_UMLS.csv", ds.UMLS(), f"results/paper_logistic_UMLS_results/evaluation_results"),
        ("datasets/UMLS_agree___with_confidence.csv", ds.UMLS(), f"results/UMLS_agree_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_appearances___with_confidence.csv", ds.UMLS(), f"results/UMLS_appearances_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_average___with_confidence.csv", ds.UMLS(), f"results/UMLS_average_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_logical___with_confidence.csv", ds.UMLS(), f"results/UMLS_logical_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_logical_with_distmult___with_confidence.csv", ds.UMLS(), f"results/UMLS_logical_with_distmult_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_model__ComplEx_with_confidence.csv", ds.UMLS(), f"results/UMLS_model_ComplEx_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_model__DistMult_with_confidence.csv", ds.UMLS(), f"results/UMLS_model_DistMult_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_model__TransE_with_confidence.csv", ds.UMLS(), f"results/results/results/UMLS_model_TransE_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_random_[0;0.5]__with_confidence.csv", ds.UMLS(), f"results/UMLS_random_with_confidences_results/evaluation_results"),
        ("datasets/UMLS_random_[0;1]__with_confidence.csv", ds.UMLS(), f"results/UMLS_random1_with_confidences_results/evaluation_results"),
        ("datasets/UMLS_random_[0.5;1]__with_confidence.csv", ds.UMLS(), f"results/UMLS_random2_with_confidences_results/evaluation_results"),
        ("datasets/UMLS_ranked_appearances___with_confidence.csv", ds.UMLS(), f"results/UMLS_ranked_appearances_with_confidence_results/evaluation_results"),
        ]
    
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

    
    