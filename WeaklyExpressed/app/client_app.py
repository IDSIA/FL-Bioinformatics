"""WeaklyExpressed: A Flower for weakly expressed genes detection."""

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from app.task import load_data, compute_low_expression_percs


# Define Flower Client
class FlowerClient(NumPyClient):

    # Initilize Flower Client
    def __init__(self, timeout, data, expr_thr):
        self.timeout = timeout
        self.data = data
        self.expr_thr = expr_thr

    # Perform local computation (perc. of gene expression values below threshold)
    def fit(self, parameters, config):
        percs = compute_low_expression_percs(self.data, self.expr_thr)
        return [percs], len(self.data), {}
        

def client_fn(context: Context):

    # Retrieve timeout (necessary for SecAggPlus)
    timeout = context.run_config["timeout"]

    # Retrieve simulation dataset parameters
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_individuals = context.run_config["num_individuals"]
    num_genes = context.run_config["num_genes"]
    seed_value = context.run_config["seed_value"]

    # Retrieve gene expression threshold
    expr_thr = context.run_config["expr_thr"]

    # Load the assigned partition of the dataset
    data = load_data(partition_id, num_partitions, num_individuals, num_genes, seed_value)
    
    return FlowerClient(timeout, data, expr_thr).to_client() 


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod, # Enable secure aggregation through SecAggPlus
    ],
)
