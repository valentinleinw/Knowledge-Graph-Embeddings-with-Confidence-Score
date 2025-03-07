import train
import time
import datetime

if __name__ == "__main__":
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_compute_TransE_with_confidence.csv", embedding_dim=50, batch_size=64, num_epochs=10, result_file=f"results/evaluation_results_{date}.csv")

    time.sleep(1)

    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_compute_DistMult_with_confidence.csv", embedding_dim=50, batch_size=64, num_epochs=10, result_file=f"results/evaluation_results_{date}.csv")

    time.sleep(1)

    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/UMLS_compute_ComplEx_with_confidence.csv", embedding_dim=50, batch_size=64, num_epochs=10, result_file=f"results/evaluation_results_{date}.csv")
