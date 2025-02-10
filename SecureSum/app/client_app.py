"""secaggexample: A Flower with SecAgg+ app."""

import time

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from app.task import get_random_vector


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self, timeout, N
    ):
        self.timeout = timeout
        self.N = N

    def fit(self, parameters, config):
        model = get_random_vector(self.N)
        return [model], 1, {}
        

def client_fn(context: Context):

    # timeout is necessary for SecAgg+
    timeout = context.run_config["timeout"]

    N = context.run_config["N"]
    print(N)
    
    return FlowerClient(timeout, N).to_client() 


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
    ],
)
