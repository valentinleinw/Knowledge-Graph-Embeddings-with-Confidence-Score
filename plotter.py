import matplotlib as plt
import csv
import pandas as pd

"""
    should have a plotter that plots:
    - same function, different dataset, singular and multiple models
    - same dataset, different function, singular and multiple models
    - multiple functions, multiple datasets, to compare the functions (only singular model I think)
"""

def plot_file(file, metric, models = []):
    
    if len(models) <= 0:
        models = ["TransE", "ComplEx", "DistMult"]
        
    models = [model + "Uncertainty" for model in models]
    
    df = pd.read_csv(file)
    
    filtered_df = df[df['Model'].isin(models)]

    x = filtered_df['Model']
    y = filtered_df[metric]

    # Create bar chart
    plt.bar(x, y)

    # Add labels and title
    plt.xlabel('Models')
    plt.ylabel(metric)
    plt.title()

    # Show the plot
    plt.show()
    
    return

def plot_different_datasets():
    return

def plot_different_datasets_and_different_functions():
    return 