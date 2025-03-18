import os
import csv

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
    def write_results_to_csv(file_name, model_name, mean_rank, mrr, hits_at_1, hits_at_5, hits_at_k, dataset_file, loss, epochs, embDims, batchSize, margin):
        # Check if the file exists to write headers only once
        file_exists = os.path.exists(file_name)

        # Open the file in append mode ('a') to add new rows instead of overwriting
        with open(file_name, mode="a", newline="") as file:
            fieldnames = ["Model", "Dataset", "Epochs", "Embedding Dims", "Batch Size", "Margin", "Loss", "Mean Rank", "MRR", "Hits@1", "Hits@5", "Hits@10"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write the header only if the file doesn't already exist
            if not file_exists:
                writer.writeheader()

            # Write the results for the given model
            writer.writerow({
                "Model": model_name,
                "Dataset": dataset_file,
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