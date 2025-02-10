"""weaklyexpr: A Flower for weakly expressed genes."""

import time
from datasets import load_dataset

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from app.task import load_data, count_alleles, compute_allele_frequencies_np, compute_allele_frequencies_np


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self, timeout, data
    ):
        self.timeout = timeout
        self.data = data

    def fit(self, parameters, config):
        allele_counts_df = count_alleles(self.data)
        allele_frequencies_np = compute_allele_frequencies_np(allele_counts_df)
        
        return [allele_frequencies_np], len(self.data), {}
        

def client_fn(context: Context):

    # timeout is necessary for SecAgg+
    timeout = context.run_config["timeout"]

    # simulation parameters
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_individuals = context.run_config["num_individuals"]
    num_snps = context.run_config["num_snps"]
    seed_value = context.run_config["seed_value"]

    data = load_data(partition_id, num_partitions, num_individuals, num_snps, seed_value)
    
    return FlowerClient(timeout, data).to_client() 


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
    ],
)
