"""WeaklyExpressed: A Flower for weakly expressed genes detection."""

import numpy as np
import pandas as pd
from datasets import Dataset

from flwr_datasets.partitioner import IidPartitioner


### Create and partition simulation dataset

def get_gene_names(num_genes: int):

    return [f"Gene_{i+1}" for i in range(num_genes)]    

def generate_gene_expression_data(num_individuals: int, num_genes: int, seed_value: int):
    """
    Generates a synthetic gene expression dataset with given parameters.
    
    Parameters:
        num_individuals (int): Number of individuals (samples).
        num_genes (int): Number of genes (features).
        seed_value (int): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: A dataframe with gene expression levels.
    """
    np.random.seed(seed_value)

    # Define gene names
    gene_names = get_gene_names(num_genes)  

    # Random mean (10-50) and std (2-10) for each gene
    mean_values = np.random.uniform(5, 50, num_genes)
    std_dev_values = np.random.uniform(2, 5, num_genes)

    # Generate gene expression levels
    data = np.zeros((num_individuals, num_genes))
    for i in range(num_genes):
        gene_data = np.abs(np.random.normal(mean_values[i], std_dev_values[i], num_individuals))
        data[:, i] = np.clip(gene_data, 0, 100)  # Clip to range [0, 100]

    # Create DataFrame
    dataset = pd.DataFrame(data, columns=gene_names)

    return dataset

partitioner = None

def load_data(partition_id: int, num_partitions: int, num_individuals: int, num_genes: int, seed_value: int):
    """
    Loads a partition of the synthetic gene expression dataset.
    
    Parameters:
        partition_id (int): The ID of the partition to load.
        num_partitions (int): Total number of partitions.
        num_individuals (int): Number of individuals in the dataset.
        num_genes (int): Number of genes in the dataset.
        seed_value (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: The selected partition of the dataset.
    """
    global partitioner
    
    if partitioner is None:  # Create partitioner only once
        np.random.seed(seed_value)
        
        dataset = generate_gene_expression_data(num_individuals, num_genes, seed_value)
        dataset = Dataset.from_pandas(dataset)

        np.random.seed(seed_value)
        partitioner = IidPartitioner(num_partitions)
        partitioner.dataset = dataset
        
    partition = partitioner.load_partition(partition_id).with_format("pandas").to_pandas()
    
    return partition


### Perform local computation (perc. of gene expression values below threshold)

def compute_expression_percs(dataset: pd.DataFrame, threshold: float):
    """
    Computes the percentage of gene expression values below a given threshold.
    
    Parameters:
        dataset (pd.DataFrame): The gene expression dataset.
        threshold (float): Expression threshold for binary classification.

    Returns:
        np.ndarray: A 1-row NumPy array containing the percentage of values below the threshold per gene.
    """
    # Create a binary matrix where expression values are below the threshold
    binary_matrix = (dataset < threshold).astype(int)
    binary_matrix.columns = dataset.columns

    # Compute percentage of values below the threshold
    total_count = binary_matrix.sum(axis=0).to_numpy().reshape(1, -1)  # Reshape to (1, num_genes)
    total_perc = total_count / len(dataset) #* 100
    
    return total_perc


### Initial conditions for FL process

def get_dummy_start():
    """
    Returns a dummy starting value for initialization.

    Returns:
        np.ndarray: A 1x1 numpy array filled with ones.
    """
    return np.ones((1, 1))


### Results of federated process
    
def get_output_df(expr_array: np.array):
    """
    Converts a NumPy expression array into a formatted DataFrame with gene names as columns.

    Args:
        expr_array (np.array): A 2D NumPy array where the first row contains gene expression values.

    Returns:
        pd.DataFrame: A DataFrame with gene names as columns and a single row of percentage-based values.
    """
    # Extract and round the first row of expression values, converting them to percentages
    values = expr_array[0].round(2)*100

    # Retrieve gene names based on the number of genes in the expression array
    gene_names = get_gene_names(values.shape[1]) 

    # Create a DataFrame with gene names as column headers
    df_expr = pd.DataFrame(values, columns=gene_names)

    return df_expr
    
def get_output_list(dataset: pd.DataFrame, threshold: int):
    """
    Extracts a list of gene names where the low-expressed percentage is greater than the given threshold.

    Args:
        dataset (pd.DataFrame): A DataFrame with gene names as columns and a single row of expression values.
        threshold (int): The threshold percentage for filtering genes.

    Returns:
        list: A list of gene names that are weakly expressed.
    """
    # Select the first row (assuming only one row in the dataset)
    values = dataset.iloc[0]

    # Filter genes where the value exceeds the threshold
    return values[values > threshold].index.tolist()

