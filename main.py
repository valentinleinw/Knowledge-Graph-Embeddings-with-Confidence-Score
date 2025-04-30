import train
from datetime import datetime
import pykeen.datasets as ds
import time

if __name__ == "__main__":
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/paper_bounded_UMLS.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/paper_logistic_UMLS.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_agree___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_appearances___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_average___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_logical___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_model__ComplEx_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_model__DistMult_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_model__TransE_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_random_[0;0.5]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_random_[0;1]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_random_[0.5;1]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_ranked_appearances___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/paper_bounded_UMLS.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/paper_logistic_UMLS.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/UMLS_agree___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/UMLS_appearances___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/UMLS_average___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/UMLS_logical___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/UMLS_model__ComplEx_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/UMLS_model__DistMult_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/UMLS_model__TransE_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/UMLS_random_[0;0.5]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/UMLS_random_[0;1]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/UMLS_random_[0.5;1]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_cosukg("datasets/UMLS_ranked_appearances___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/paper_bounded_UMLS.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/paper_logistic_UMLS.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/UMLS_agree___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/UMLS_appearances___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/UMLS_average___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/UMLS_logical___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/UMLS_model__ComplEx_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/UMLS_model__DistMult_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/UMLS_model__TransE_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/UMLS_random_[0;0.5]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/UMLS_random_[0;1]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/UMLS_random_[0.5;1]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_inverse("datasets/UMLS_ranked_appearances___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/paper_bounded_UMLS.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/paper_logistic_UMLS.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/UMLS_agree___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/UMLS_appearances___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/UMLS_average___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/UMLS_logical___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/UMLS_model__ComplEx_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/UMLS_model__DistMult_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/UMLS_model__TransE_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/UMLS_random_[0;0.5]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/UMLS_random_[0;1]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/UMLS_random_[0.5;1]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_neg_confidences_similarity("datasets/UMLS_ranked_appearances___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/paper_bounded_UMLS.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/paper_logistic_UMLS.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/UMLS_agree___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/UMLS_appearances___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/UMLS_average___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/UMLS_logical___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/UMLS_model__ComplEx_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/UMLS_model__DistMult_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/UMLS_model__TransE_with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/UMLS_random_[0;0.5]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/UMLS_random_[0;1]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/UMLS_random_[0.5;1]__with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate_objective_function("datasets/UMLS_ranked_appearances___with_confidence.csv", ds.UMLS(), embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/paper_bounded_UMLS.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/paper_logistic_UMLS.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/UMLS_agree___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/UMLS_appearances___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/UMLS_average___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/UMLS_logical___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/UMLS_model__ComplEx_with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/UMLS_model__DistMult_with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/UMLS_model__TransE_with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/UMLS_random_[0;0.5]__with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/UMLS_random_[0;1]__with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/UMLS_random_[0.5;1]__with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_transE_with_different_losses("datasets/UMLS_ranked_appearances___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/paper_bounded_UMLS.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/paper_logistic_UMLS.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/UMLS_agree___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/UMLS_appearances___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/UMLS_average___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/UMLS_logical___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/UMLS_model__ComplEx_with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/UMLS_model__DistMult_with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/UMLS_model__TransE_with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/UMLS_random_[0;0.5]__with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/UMLS_random_[0;1]__with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/UMLS_random_[0.5;1]__with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_distmult_with_different_losses("datasets/UMLS_ranked_appearances___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/paper_bounded_UMLS.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/paper_logistic_UMLS.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/UMLS_agree___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/UMLS_appearances___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/UMLS_average___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/UMLS_logical___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/UMLS_model__ComplEx_with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/UMLS_model__DistMult_with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/UMLS_model__TransE_with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/UMLS_random_[0;0.5]__with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/UMLS_random_[0;1]__with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/UMLS_random_[0.5;1]__with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_complex_with_different_losses("datasets/UMLS_ranked_appearances___with_confidence.csv", embedding_dim=50, batch_size=512, num_epochs=100, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    