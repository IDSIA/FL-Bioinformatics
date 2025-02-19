"""MinorAllele: A Flower for minor allele frequency."""

from logging import DEBUG, INFO
import numpy as np

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, log
from flwr.common.logger import update_console_handler
import flwr.common.recordset_compat as compat
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD
from flwr.server.strategy import FedAvg

from app.task import get_dummy_start, compute_maf, get_SNP_names, create_out_df


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

    # Store final results in a file
    paramsrecord = context.state.parameters_records[MAIN_PARAMS_RECORD]
    parameters = compat.parametersrecord_to_parameters(paramsrecord, True)
    ndarrays = parameters_to_ndarrays(parameters)

    # Ensure values is a NumPy array
    values = np.array(ndarrays[0])
    #print(values)

    # Create the output
    out = compute_maf(values)
    names = get_SNP_names(len(out))
    out_df = create_out_df(names,out)
    out_df.to_csv('maf_out.csv')

    log(INFO, "")
    log(
        INFO,
        "################################ Final output ################################",
    )
    
    # Format the output to show the first two digits for each MAF value
    formatted_out = [f"{name}, {round(maf,2)}" for name,maf in zip(names,out)]
    
    log(
        INFO,
        "Minor allele frequencies: %s",
        ", ".join(formatted_out),  # Join the formatted MAF values with a comma and space
    )
    
    log(INFO, "")