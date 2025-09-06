import train
import pykeen.datasets as ds
import pandas as pd

if __name__ == "__main__":
    
    def avg(csv_path):
        df = pd.read_csv(csv_path)

        # Convert 'N/A' and other non-numeric entries to NaN (for averaging)
        df.replace("N/A", pd.NA, inplace=True)

        # Columns that define uniqueness of a configuration
        group_cols = ["Model"]

        # Group by these and average the rest
        df_avg = df.groupby("Model", as_index=False).agg({
            **{col: "mean" for col in df.select_dtypes(include="number").columns},
            "Function": "first"   # keep the Function value
        })

        # Save back to the same file, overwriting the original
        df_avg.to_csv(csv_path, index=False)
    
    """triples = [
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
        
        ("datasets/paper_bounded_CoDExLarge.csv", ds.CoDExLarge(), f"results/paper_bounded_CoDExLarge_results/evaluation_results"),
        ("datasets/paper_logistic_CoDExLarge.csv", ds.CoDExLarge(), f"results/paper_logistic_CoDExLarge_results/evaluation_results"),
        ("datasets/CoDExLarge_agree___with_confidence.csv", ds.CoDExLarge(), f"results/CoDExLarge_agree_with_confidence_results/evaluation_results"),
        ("datasets/CoDExLarge_appearances___with_confidence.csv", ds.CoDExLarge(), f"results/CoDExLarge_appearances_with_confidence_results/evaluation_results"),
        ("datasets/CoDExLarge_average___with_confidence.csv", ds.CoDExLarge(), f"results/CoDExLarge_average_with_confidence_results/evaluation_results"),
        ("datasets/CoDExLarge_logical___with_confidence.csv", ds.CoDExLarge(), f"results/CoDExLarge_logical_with_confidence_results/evaluation_results"),
        ("datasets/CoDExLarge_logical_with_distmult___with_confidence.csv", ds.CoDExLarge(), f"results/CoDExLarge_logical_with_distmult_with_confidence_results/evaluation_results"),
        ("datasets/CoDExLarge_model__ComplEx_with_confidence.csv", ds.CoDExLarge(), f"results/CoDExLarge_model_ComplEx_with_confidence_results/evaluation_results"),
        ("datasets/CoDExLarge_model__DistMult_with_confidence.csv", ds.CoDExLarge(), f"results/CoDExLarge_model_DistMult_with_confidence_results/evaluation_results"),
        ("datasets/CoDExLarge_model__TransE_with_confidence.csv", ds.CoDExLarge(), f"results/CoDExLarge_model_TransE_with_confidence_results/evaluation_results"),
        ("datasets/CoDExLarge_random_[0;0.5]__with_confidence.csv", ds.CoDExLarge(), f"results/CoDExLarge_random_with_confidences_results/evaluation_results"),
        ("datasets/CoDExLarge_random_[0;1]__with_confidence.csv", ds.CoDExLarge(), f"results/CoDExLarge_random1_with_confidences_results/evaluation_results"),
        ("datasets/CoDExLarge_random_[0.5;1]__with_confidence.csv", ds.CoDExLarge(), f"results/CoDExLarge_random2_with_confidences_results/evaluation_results"),
        ("datasets/CoDExLarge_ranked_appearances___with_confidence.csv", ds.CoDExLarge(), f"results/CoDExLarge_ranked_appearances_with_confidence_results/evaluation_results"),
        
        ("datasets/paper_bounded_YAGO310.csv", ds.YAGO310(), f"results/paper_bounded_YAGO310_results/evaluation_results"),
        ("datasets/paper_logistic_YAGO310.csv", ds.YAGO310(), f"results/paper_logistic_YAGO310_results/evaluation_results"),
        ("datasets/YAGO310_agree___with_confidence.csv", ds.YAGO310(), f"results/YAGO310_agree_with_confidence_results/evaluation_results"),
        ("datasets/YAGO310_appearances___with_confidence.csv", ds.YAGO310(), f"results/YAGO310_appearances_with_confidence_results/evaluation_results"),
        ("datasets/YAGO310_average___with_confidence.csv", ds.YAGO310(), f"results/YAGO310_average_with_confidence_results/evaluation_results"),
        ("datasets/YAGO310_logical___with_confidence.csv", ds.YAGO310(), f"results/YAGO310_logical_with_confidence_results/evaluation_results"),
        ("datasets/YAGO310_logical_with_distmult___with_confidence.csv", ds.YAGO310(), f"results/YAGO310_logical_with_distmult_with_confidence_results/evaluation_results"),
        ("datasets/YAGO310_model__ComplEx_with_confidence.csv", ds.YAGO310(), f"results/YAGO310_model_ComplEx_with_confidence_results/evaluation_results"),
        ("datasets/YAGO310_model__DistMult_with_confidence.csv", ds.YAGO310(), f"results/YAGO310_model_DistMult_with_confidence_results/evaluation_results"),
        ("datasets/YAGO310_model__TransE_with_confidence.csv", ds.YAGO310(), f"results/YAGO310_model_TransE_with_confidence_results/evaluation_results"),
        ("datasets/YAGO310_random_[0;0.5]__with_confidence.csv", ds.YAGO310(), f"results/YAGO310_random_with_confidences_results/evaluation_results"),
        ("datasets/YAGO310_random_[0;1]__with_confidence.csv", ds.YAGO310(), f"results/YAGO310_random1_with_confidences_results/evaluation_results"),
        ("datasets/YAGO310_random_[0.5;1]__with_confidence.csv", ds.YAGO310(), f"results/YAGO310_random2_with_confidences_results/evaluation_results"),
        ("datasets/YAGO310_ranked_appearances___with_confidence.csv", ds.YAGO310(), f"results/YAGO310_ranked_appearances_with_confidence_results/evaluation_results")]"""
    
    #for origin, dataset, result in triples:
    origin = "datasets/CoDExSmall_model__DistMult_with_confidence.csv"
    dataset = ds.CoDExSmall()
    result = "results/CoDExSmall_model_DistMult_with_confidence_results/evaluation_results"
    for i in range(10):
        train.train_and_evaluate(origin, dataset, "loss", embedding_dim=100, batch_size=1024, num_epochs=500, result_file=result + "_loss.csv")
        
        train.train_and_evaluate(origin, dataset, "objective", embedding_dim=100, batch_size=1024, num_epochs=500, result_file=result + "_objective.csv")

        train.train_and_evaluate(origin, dataset, "divergence", embedding_dim=100, batch_size=1024, num_epochs=500, result_file=result + "_divergence.csv")

        train.train_and_evaluate(origin, dataset, "gaussian", embedding_dim=100, batch_size=1024, num_epochs=500, result_file=result + "_gaussian.csv")

        train.train_and_evaluate(origin, dataset, "softplus", embedding_dim=100, batch_size=1024, num_epochs=500, result_file=result + "_softplus.csv")
        
        train.train_and_evaluate_neg_confidences_cosukg(origin, dataset, embedding_dim=100, batch_size=1024, num_epochs=500, result_file=result + "_cosukg.csv")

        train.train_and_evaluate_neg_confidences_inverse(origin, dataset, embedding_dim=100, batch_size=1024, num_epochs=500, result_file=result + "_inverse.csv")

        train.train_and_evaluate_neg_confidences_similarity(origin, dataset, embedding_dim=100, batch_size=1024, num_epochs=500, result_file=result + "_similarity.csv")
        
        print("Finished iteration number " + str(i))

    avg(result + "_loss.csv")
    
    avg(result + "_objective.csv")
    
    avg(result + "_divergence.csv")
    
    avg(result + "_gaussian.csv")
    
    avg(result + "_softplus.csv")
    
    avg(result + "_cosukg.csv")
    
    avg(result + "_inverse.csv")
    
    avg(result + "_similarity.csv")
    
print("Finished!")
    
    