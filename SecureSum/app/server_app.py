"""SecureSum: A Flower for custom secure sum strategy using SecAgg+."""

from logging import DEBUG

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import update_console_handler
from flwr.common.record import ParametersRecord
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD
from flwr.server.strategy import FedAvg

from app.task import get_dummy_start
from app.custom_strategy import FedSum


# Flower ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:

    # Define strategy
    strategy = FedSum(
        # Use all clients
        fraction_fit=1.0,
        # Interrupt if any client fails
        accept_failures=False,
        # Disable evaluation
        fraction_evaluate=0.0,
        # Dummy initial conditions for sum
        initial_parameters=ndarrays_to_parameters([get_dummy_start()]),
    )

    # Construct the LegacyContext
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=1),
        strategy=strategy,
    )

    # Create fit workflow
    # For further information, please see:
    # https://flower.ai/docs/framework/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html
    update_console_handler(DEBUG, True, True)
    fit_workflow = SecAggPlusWorkflow(
        num_shares=context.run_config["num-shares"],
        reconstruction_threshold=context.run_config["reconstruction-threshold"],
        timeout=context.run_config["timeout"],
    )

    # Create the workflow
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    # Execute
    workflow(grid, context)

    # Final result - printed in the logs (see workflow file)
    paramsrecord = context.state[MAIN_PARAMS_RECORD]
    ndarrays = ParametersRecord.to_numpy_ndarrays(paramsrecord)
