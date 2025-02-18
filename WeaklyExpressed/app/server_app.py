"""WeaklyExpressed: A Flower for weakly expressed genes detection."""

from logging import DEBUG, INFO
import numpy as np

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, log
from flwr.common.logger import update_console_handler
import flwr.common.recordset_compat as compat
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD
from flwr.server.strategy import FedAvg

from app.task import get_dummy_start, get_gene_names, get_output_df, get_output_list


# Flower ServerApp
app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:

    # Define strategy
    strategy = FedAvg(
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
    # For further information on SecAggPlus with Flower, please refer to:
    # https://flower.ai/docs/examples/flower-secure-aggregation.html
    update_console_handler(DEBUG, True, True)
    fit_workflow = SecAggPlusWorkflow(
        num_shares=context.run_config["num-shares"],
        reconstruction_threshold=context.run_config["reconstruction-threshold"],
        timeout=context.run_config["timeout"],
    )

    # Create the workflow
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    # Execute
    workflow(driver, context)

    # Extract the final result
    paramsrecord = context.state.parameters_records[MAIN_PARAMS_RECORD]
    parameters = compat.parametersrecord_to_parameters(paramsrecord, True)
    ndarrays = parameters_to_ndarrays(parameters)
    np.save("output.npy", ndarrays)
    
    # Prepare screen output
    df_out = get_output_df(ndarrays)
    list_out = get_output_list(df_out, context.run_config["expr_perc"])

    # Print output
    log(INFO, "")
    log(
        INFO,
        "################################ Final output ################################",
    )
    log(INFO, "")
    log(INFO, "Percentage of samples where gene is expressed less than threshold:\n%s", 
        df_out.to_string(index=False))
    log(INFO, "")
    log(INFO, "Weakly expressed genes: %s", list_out)
    log(INFO, "")


