"""Aggregation functions for strategy implementations."""

from functools import partial, reduce

import numpy as np

from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


def aggregate_sum(results: list[tuple[NDArrays, int]]) -> NDArrays:
    """Compute the sum of weights."""
    # Create a list of weights
    summed_weights = [
        [layer for layer in weights] for weights, _ in results
    ]

    # Compute the sum of weights for each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates)
        for layer_updates in zip(*summed_weights)
    ]
    return weights_prime

def aggregate_inplace_sum(results: list[tuple[ClientProxy, FitRes]]) -> NDArrays:
    """Compute in-place sum of parameters."""
    def _try_inplace(
        x: NDArrays, y: NDArrays, np_binary_op: np.ufunc
    ) -> NDArrays:
        return (  # type: ignore[no-any-return]
            np_binary_op(x, y, out=x)
            if np.can_cast(y, x.dtype, casting="same_kind")
            else np_binary_op(x, np.array(y, x.dtype), out=x)
        )

    # Let's do in-place aggregation (sum)
    # Get first result, then add up each other
    params = parameters_to_ndarrays(results[0][1].parameters)

    for _, fit_res in results[1:]:
        res = parameters_to_ndarrays(fit_res.parameters)
        params = [
            reduce(partial(_try_inplace, np_binary_op=np.add), layer_updates)
            for layer_updates in zip(params, res)
        ]

    return params

