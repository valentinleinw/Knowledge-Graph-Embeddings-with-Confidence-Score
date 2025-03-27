import train
from datetime import datetime
import pykeen.datasets as ds
import time

if __name__ == "__main__":
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/paper_bounded_UMLS.csv", ds.UMLS(), embedding_dim=6, batch_size=16, num_epochs=5, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/paper_logistic_UMLS.csv", ds.UMLS(), embedding_dim=6, batch_size=16, num_epochs=5, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/paper_bounded_UMLS.csv", ds.UMLS(), embedding_dim=6, batch_size=16, num_epochs=5, result_file=f"results/evaluation_results_{date}.csv")
    
    time.sleep(1)
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    train.train_and_evaluate("datasets/paper_logistic_UMLS.csv", ds.UMLS(), embedding_dim=6, batch_size=16, num_epochs=5, result_file=f"results/evaluation_results_{date}.csv")
    
