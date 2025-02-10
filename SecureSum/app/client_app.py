"""SecureSum: A Flower for custom secure sum strategy using SecAgg+."""

import time

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from app.task import get_random_vector


# Define Flower Client
class FlowerClient(NumPyClient):

    # Initilize Flower Client
    def __init__(self, timeout, N):
        self.timeout = timeout
        self.N = N

    # Generate a random N-component vector locally
    def fit(self, parameters, config):
        model = get_random_vector(self.N)
        return [model], 1, {}
        

def client_fn(context: Context):

    # Retrieve timeout (necessary for SecAgg+)
    timeout = context.run_config["timeout"]

    # Retrieve the number of components of the vectors to be summed
    N = context.run_config["N"]
    
    return FlowerClient(timeout, N).to_client() 


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod, # Enable secure aggregation through SecAgg+
    ],
)
