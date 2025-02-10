"""secaggexample: A Flower with SecAgg+ app."""

from logging import DEBUG
import csv 

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import update_console_handler
import flwr.common.recordset_compat as compat
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD

from app.task import get_dummy_start
from app.workflow_with_log import SecAggPlusWorkflowWithLogs
from app.custom_strategy import FedSum


# Flower ServerApp
app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:

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
    fit_workflow = SecAggPlusWorkflowWithLogs(
        num_shares=context.run_config["num-shares"],
        reconstruction_threshold=context.run_config["reconstruction-threshold"],
        max_weight=1,
        timeout=context.run_config["timeout"],
    )

    # Create the workflow
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    # Execute
    workflow(driver, context)

    # Store final results in a file
    paramsrecord = context.state.parameters_records[MAIN_PARAMS_RECORD]
    parameters = compat.parametersrecord_to_parameters(paramsrecord, True)
    ndarrays = parameters_to_ndarrays(parameters)
    with open("output.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(ndarrays)


    