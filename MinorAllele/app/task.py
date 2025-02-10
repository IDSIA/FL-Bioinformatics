"""weaklyexpr: A Flower for weakly expressed genes."""

import random
import numpy as np
import pandas as pd
from datasets import Dataset
from collections import Counter

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

def get_dummy_start():
    """
    Returns a dummy starting value for initialization.

    Returns:
        np.ndarray: A 1x1 numpy array filled with ones.
    """
    return np.ones((1, 1))

def generate_gwas_dataset(num_individuals: int, num_snps: int, seed_value: int):
    """
    Generates a synthetic GWAS dataset with random SNP genotypes for multiple alleles.

    Parameters:
        num_individuals (int): Number of individuals (rows in dataset).
        num_snps (int): Number of SNPs (columns in dataset).
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A GWAS dataset where each cell represents a genotype based on the provided alleles.
    """
    np.random.seed(seed_value)
    
    # Generate all possible genotype combinations from the given alleles
    alleles = ['A', 'G', 'C', 'T']
    genotypes = [f"{a}{b}" for i, a in enumerate(alleles) for b in alleles[i:]]  # This creates combinations like AA, AG, GG, etc.
    
    # Generate random SNP genotypes for each individual
    data = np.random.choice(genotypes, size=(num_individuals, num_snps))
    
    # Create DataFrame with SNP names
    snp_names = [f"SNP_{i+1}" for i in range(num_snps)]
    individual_ids = [f"Ind_{i+1}" for i in range(num_individuals)]
    
    df = pd.DataFrame(data, columns=snp_names, index=individual_ids)
    
    return df

def count_alleles(gwas_df: pd.DataFrame):
    """
    Counts the occurrences of each allele (A and G) for every SNP in the dataset.

    Parameters:
        gwas_df (pd.DataFrame): A DataFrame where each cell contains a genotype ('AA', 'AG', or 'GG').

    Returns:
        pd.DataFrame: A DataFrame with allele counts for each SNP.
    """
    allele_counts = {}

    for snp in gwas_df.columns:
        # Flatten the list of alleles in all genotypes for this SNP
        all_alleles = ''.join(gwas_df[snp].values)  # Merge all genotypes into one string
        allele_count = Counter(all_alleles)  # Count occurrences of 'A' and 'G'
        
        # Store counts in dictionary
        allele_counts[snp] = {
            'A_count': allele_count.get('A', 0),
            'C_count': allele_count.get('C', 0),
            'G_count': allele_count.get('G', 0),
            'T_count': allele_count.get('T', 0),
            'Total_Alleles': allele_count.get('A', 0) + allele_count.get('C', 0) + allele_count.get('G', 0) + allele_count.get('T', 0)
        }

    return pd.DataFrame.from_dict(allele_counts, orient='index')

def compute_allele_frequencies_np(allele_counts_df: pd.DataFrame):
    """
    Computes the frequency of each allele (A and G) for every SNP and returns as a NumPy array.

    Parameters:
        allele_counts_df (pd.DataFrame): A DataFrame with 'A_count', 'G_count', and 'Total_Alleles' per SNP.

    Returns:
        np.ndarray: A 2D NumPy array where each row represents [A_frequency, G_frequency] for a SNP.
    """
    # Compute allele frequencies
    A_frequency = allele_counts_df["A_count"] / allele_counts_df["Total_Alleles"]
    C_frequency = allele_counts_df["C_count"] / allele_counts_df["Total_Alleles"]
    G_frequency = allele_counts_df["G_count"] / allele_counts_df["Total_Alleles"]
    T_frequency = allele_counts_df["T_count"] / allele_counts_df["Total_Alleles"]

    # Convert to NumPy array
    return np.column_stack((A_frequency, C_frequency, G_frequency, T_frequency))

partitioner = None

def load_data(partition_id: int, num_partitions: int, num_individuals: int, num_snps: int, seed_value: int):
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
        
        dataset = generate_gwas_dataset(num_individuals, num_snps, seed_value)
        dataset = Dataset.from_pandas(dataset)

        np.random.seed(seed_value)
        partitioner = IidPartitioner(num_partitions)
        partitioner.dataset = dataset
        
    partition = partitioner.load_partition(partition_id).with_format("pandas").to_pandas()
    
    return partition

def compute_maf(allele_frequencies_np: np.ndarray):
    """
    Computes the Minor Allele Frequency (MAF) for each SNP.

    Parameters:
        allele_frequencies_np (np.ndarray): A 2D NumPy array where each row contains [A_frequency, G_frequency].

    Returns:
        np.ndarray: A 1D NumPy array containing the MAF for each SNP.
    """
    # MAF is the minimum of the two allele frequencies for each SNP
    maf = np.min(allele_frequencies_np, axis=1)
    return maf
