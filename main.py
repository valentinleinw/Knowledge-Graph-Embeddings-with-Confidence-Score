import train
from datetime import datetime
import pykeen.datasets as ds
import time

if __name__ == "__main__":
    
    triples = [
        ("datasets/paper_bounded_UMLS.csv", ds.UMLS(), f"results/paper_bounded_UMLS_results/evaluation_results"),
        ("datasets/paper_logistic_UMLS.csv", ds.UMLS(), f"results/paper_logistic_UMLS_results/evaluation_results"),
        ("datasets/UMLS_agree___with_confidence.csv", ds.UMLS(), f"results/UMLS_agree_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_appearances___with_confidence.csv", ds.UMLS(), f"results/UMLS_appearances_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_average___with_confidence.csv", ds.UMLS(), f"results/UMLS_average_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_logical___with_confidence.csv", ds.UMLS(), f"results/UMLS_logical_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_logical_with_distmult___with_confidence.csv", ds.UMLS(), f"results/UMLS_logical_with_distmult_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_model__ComplEx_with_confidence.csv", ds.UMLS(), f"results/UMLS_model_ComplEx_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_model__DistMult_with_confidence.csv", ds.UMLS(), f"results/UMLS_model_DistMult_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_model__TransE_with_confidence.csv", ds.UMLS(), f"results/UMLS_model_TransE_with_confidence_results/evaluation_results"),
        ("datasets/UMLS_random_[0;0.5]__with_confidence.csv", ds.UMLS(), f"results/UMLS_random_with_confidences_results/evaluation_results"),
        ("datasets/UMLS_random_[0;1]__with_confidence.csv", ds.UMLS(), f"results/UMLS_random1_with_confidences_results/evaluation_results"),
        ("datasets/UMLS_random_[0.5;1]__with_confidence.csv", ds.UMLS(), f"results/UMLS_random2_with_confidences_results/evaluation_results"),
        ("datasets/UMLS_ranked_appearances___with_confidence.csv", ds.UMLS(), f"results/UMLS_ranked_appearances_with_confidence_results/evaluation_results"),
        
        ("datasets/paper_bounded_WN18RR.csv", ds.WN18RR(), f"results/paper_bounded_WN18RR_results/evaluation_results"),
        ("datasets/paper_logistic_WN18RR.csv", ds.WN18RR(), f"results/paper_logistic_WN18RR_results/evaluation_results"),
        ("datasets/WN18RR_agree___with_confidence.csv", ds.WN18RR(), f"results/WN18RR_agree_with_confidence_results/evaluation_results"),
        ("datasets/WN18RR_appearances___with_confidence.csv", ds.WN18RR(), f"results/WN18RR_appearances_with_confidence_results/evaluation_results"),
        ("datasets/WN18RR_average___with_confidence.csv", ds.WN18RR(), f"results/WN18RR_average_with_confidence_results/evaluation_results"),
        ("datasets/WN18RR_logical___with_confidence.csv", ds.WN18RR(), f"results/WN18RR_logical_with_confidence_results/evaluation_results"),
        ("datasets/WN18RR_logical_with_distmult___with_confidence.csv", ds.WN18RR(), f"results/WN18RR_logical_with_distmult_with_confidence_results/evaluation_results"),
        ("datasets/WN18RR_model__ComplEx_with_confidence.csv", ds.WN18RR(), f"results/WN18RR_model_ComplEx_with_confidence_results/evaluation_results"),
        ("datasets/WN18RR_model__DistMult_with_confidence.csv", ds.WN18RR(), f"results/WN18RR_model_DistMult_with_confidence_results/evaluation_results"),
        ("datasets/WN18RR_model__TransE_with_confidence.csv", ds.WN18RR(), f"results/WN18RR_model_TransE_with_confidence_results/evaluation_results"),
        ("datasets/WN18RR_random_[0;0.5]__with_confidence.csv", ds.WN18RR(), f"results/WN18RR_random_with_confidences_results/evaluation_results"),
        ("datasets/WN18RR_random_[0;1]__with_confidence.csv", ds.WN18RR(), f"results/WN18RR_random1_with_confidences_results/evaluation_results"),
        ("datasets/WN18RR_random_[0.5;1]__with_confidence.csv", ds.WN18RR(), f"results/WN18RR_random2_with_confidences_results/evaluation_results"),
        ("datasets/WN18RR_ranked_appearances___with_confidence.csv", ds.WN18RR(), f"results/WN18RR_ranked_appearances_with_confidence_results/evaluation_results"),
        
        ("datasets/paper_bounded_CoDExSmall.csv", ds.CoDExSmall(), f"results/paper_bounded_CoDExSmall_results/evaluation_results"),
        ("datasets/paper_logistic_CoDExSmall.csv", ds.CoDExSmall(), f"results/paper_logistic_CoDExSmall_results/evaluation_results"),
        ("datasets/CoDExSmall_agree___with_confidence.csv", ds.CoDExSmall(), f"results/CoDExSmall_agree_with_confidence_results/evaluation_results"),
        ("datasets/CoDExSmall_appearances___with_confidence.csv", ds.CoDExSmall(), f"results/CoDExSmall_appearances_with_confidence_results/evaluation_results"),
        ("datasets/CoDExSmall_average___with_confidence.csv", ds.CoDExSmall(), f"results/CoDExSmall_average_with_confidence_results/evaluation_results"),
        ("datasets/CoDExSmall_logical___with_confidence.csv", ds.CoDExSmall(), f"results/CoDExSmall_logical_with_confidence_results/evaluation_results"),
        ("datasets/CoDExSmall_logical_with_distmult___with_confidence.csv", ds.CoDExSmall(), f"results/CoDExSmall_logical_with_distmult_with_confidence_results/evaluation_results"),
        ("datasets/CoDExSmall_model__ComplEx_with_confidence.csv", ds.CoDExSmall(), f"results/CoDExSmall_model_ComplEx_with_confidence_results/evaluation_results"),
        ("datasets/CoDExSmall_model__DistMult_with_confidence.csv", ds.CoDExSmall(), f"results/CoDExSmall_model_DistMult_with_confidence_results/evaluation_results"),
        ("datasets/CoDExSmall_model__TransE_with_confidence.csv", ds.CoDExSmall(), f"results/CoDExSmall_model_TransE_with_confidence_results/evaluation_results"),
        ("datasets/CoDExSmall_random_[0;0.5]__with_confidence.csv", ds.CoDExSmall(), f"results/CoDExSmall_random_with_confidences_results/evaluation_results"),
        ("datasets/CoDExSmall_random_[0;1]__with_confidence.csv", ds.CoDExSmall(), f"results/CoDExSmall_random1_with_confidences_results/evaluation_results"),
        ("datasets/CoDExSmall_random_[0.5;1]__with_confidence.csv", ds.CoDExSmall(), f"results/CoDExSmall_random2_with_confidences_results/evaluation_results"),
        ("datasets/CoDExSmall_ranked_appearances___with_confidence.csv", ds.CoDExSmall(), f"results/CoDExSmall_ranked_appearances_with_confidence_results/evaluation_results"),
        
        ("datasets/paper_bounded_CoDExMedium.csv", ds.CoDExMedium(), f"results/paper_bounded_CoDExMedium_results/evaluation_results"),
        ("datasets/paper_logistic_CoDExMedium.csv", ds.CoDExMedium(), f"results/paper_logistic_CoDExMedium_results/evaluation_results"),
        ("datasets/CoDExMedium_agree___with_confidence.csv", ds.CoDExMedium(), f"results/CoDExMedium_agree_with_confidence_results/evaluation_results"),
        ("datasets/CoDExMedium_appearances___with_confidence.csv", ds.CoDExMedium(), f"results/CoDExMedium_appearances_with_confidence_results/evaluation_results"),
        ("datasets/CoDExMedium_average___with_confidence.csv", ds.CoDExMedium(), f"results/CoDExMedium_average_with_confidence_results/evaluation_results"),
        ("datasets/CoDExMedium_logical___with_confidence.csv", ds.CoDExMedium(), f"results/CoDExMedium_logical_with_confidence_results/evaluation_results"),
        ("datasets/CoDExMedium_logical_with_distmult___with_confidence.csv", ds.CoDExMedium(), f"results/CoDExMedium_logical_with_distmult_with_confidence_results/evaluation_results"),
        ("datasets/CoDExMedium_model__ComplEx_with_confidence.csv", ds.CoDExMedium(), f"results/CoDExMedium_model_ComplEx_with_confidence_results/evaluation_results"),
        ("datasets/CoDExMedium_model__DistMult_with_confidence.csv", ds.CoDExMedium(), f"results/CoDExMedium_model_DistMult_with_confidence_results/evaluation_results"),
        ("datasets/CoDExMedium_model__TransE_with_confidence.csv", ds.CoDExMedium(), f"results/CoDExMedium_model_TransE_with_confidence_results/evaluation_results"),
        ("datasets/CoDExMedium_random_[0;0.5]__with_confidence.csv", ds.CoDExMedium(), f"results/CoDExMedium_random_with_confidences_results/evaluation_results"),
        ("datasets/CoDExMedium_random_[0;1]__with_confidence.csv", ds.CoDExMedium(), f"results/CoDExMedium_random1_with_confidences_results/evaluation_results"),
        ("datasets/CoDExMedium_random_[0.5;1]__with_confidence.csv", ds.CoDExMedium(), f"results/CoDExMedium_random2_with_confidences_results/evaluation_results"),
        ("datasets/CoDExMedium_ranked_appearances___with_confidence.csv", ds.CoDExMedium(), f"results/CoDExMedium_ranked_appearances_with_confidence_results/evaluation_results"),
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

    
    