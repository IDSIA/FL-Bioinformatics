"""WeaklyExpressed: A Flower for weakly expressed genes detection.

This file is based on Flower's SecAgg+ tutorial 
(https://flower.ai/docs/examples/flower-secure-aggregation.html)
"""

from logging import INFO

import flwr.common.recordset_compat as compat
from flwr.common import Context, log, parameters_to_ndarrays
from flwr.common.secure_aggregation.quantization import quantize
from flwr.server import Driver, LegacyContext
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD
from flwr.server.workflow.secure_aggregation.secaggplus_workflow import (
    SecAggPlusWorkflow,
    WorkflowState,
)


class SecAggPlusWorkflowWithLogs(SecAggPlusWorkflow):

    node_ids = []

    def __call__(self, driver: Driver, context: Context) -> None:
        log(INFO, "")
        log(
            INFO,
            "########################## Secure Aggregation Start ##########################",
        )

        super().__call__(driver, context)

        paramsrecord = context.state.parameters_records[MAIN_PARAMS_RECORD]
        parameters = compat.parametersrecord_to_parameters(paramsrecord, True)
        ndarrays = parameters_to_ndarrays(parameters)
        
        log(INFO, "")
        log(
            INFO,
            "########################### Secure Aggregation End ###########################",
        )
        log(INFO, "")

    def setup_stage(
        self, driver: Driver, context: LegacyContext, state: WorkflowState
    ) -> bool:
        ret = super().setup_stage(driver, context, state)
        self.node_ids = list(state.active_node_ids)
        state.nid_to_fitins[self.node_ids[0]].configs_records["fitins.config"][
            "drop"
        ] = True
        return ret

    def collect_masked_vectors_stage(
        self, driver: Driver, context: LegacyContext, state: WorkflowState
    ) -> bool:
        ret = super().collect_masked_vectors_stage(driver, context, state)
        for node_id in state.sampled_node_ids - state.active_node_ids:
            log(INFO, "Client %s dropped out.", self.node_ids.index(node_id))
        return ret
