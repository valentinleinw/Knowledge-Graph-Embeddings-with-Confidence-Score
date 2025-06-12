import os
import csv
import pandas as pd

class csvEditor():
    @staticmethod
    def save_to_csv(df, dataset, method_name, model="", range=""):
        dataset_name = dataset.__class__.__name__  # Get dataset name dynamically
        
        # Define file path for saving
        folder_path = "datasets"
        file_path = os.path.join(folder_path, f"{dataset_name}_{method_name}_{range}_{model}_with_confidence.csv")

        # Ensure the directory exists
        os.makedirs(folder_path, exist_ok=True)

        # Save DataFrame to CSV
        df.to_csv(file_path, index=False)

        print(f"Dataset with confidence scores saved successfully at: {file_path}")
        
    @staticmethod
    def write_results_to_csv(file_name, subfolder_name, function_name, model_name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k, dataset_file, loss, epochs, embDims, batchSize, margin):
        
        os.makedirs(subfolder_name, exist_ok=True)
        
        full_file_path = os.path.join(subfolder_name, file_name)
        
        # Check if the file exists to write headers only once
        file_exists = os.path.exists(full_file_path)

        # Open the file in append mode ('a') to add new rows instead of overwriting
        with open(full_file_path, mode="a", newline="") as file:
            fieldnames = ["Model", "Dataset", "Function", "Epochs", "Embedding Dims", "Batch Size", "Margin", "Loss", "Mean Rank", "MRR", "Hits@1", "Hits@5", "Hits@10"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write the header only if the file doesn't already exist
            if not file_exists:
                writer.writeheader()

            # Write the results for the given model
            writer.writerow({
                "Model": model_name,
                "Dataset": dataset_file,
                "Function": function_name,
                "Epochs": epochs,
                "Embedding Dims": embDims,
                "Batch Size": batchSize,
                "Margin": margin,
                "Loss": loss,
                "Mean Rank": mean_rank,
                "MRR": mrr,
                "Hits@1": hits_at_1,
                "Hits@5": hits_at_5,
                "Hits@10": hits_at_k,
            })
            
    @staticmethod
    def save_to_csv_paper(dataset, confidence_scores, triples, approach):
        confidence_scores = confidence_scores.detach().cpu().numpy()

        # Prepare data for CSV
        data = []
        for i, (head, relation, tail) in enumerate(triples.tolist()):
            data.append([
                head,  
                relation,  
                tail,  
                float(confidence_scores[i]) 
            ])

        
        df = pd.DataFrame(data, columns=["head", "relation", "tail", "confidence_score"])
        
        dataset_name = dataset.__class__.__name__
        folder_path = "datasets"
        file_path = os.path.join(folder_path, f"paper_{approach}_{dataset_name}.csv")
        
        os.makedirs(folder_path, exist_ok=True)

        
        df.to_csv(file_path, index=False)