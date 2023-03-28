#date: 2023-03-28T16:45:25Z
#url: https://api.github.com/gists/cfe09f9464a9f90d7db2cefade167222
#owner: https://api.github.com/users/sherry-x

from spec_types.alert_utils import (
    create_alert_group,
    create_group_by_trigger,
    get_cluster,
    AlertMetric,
    MAINNET_SERVICE,
    PREVIEWNET_SERVICE,
)
from spec_types.lib_fix_vour import (
    AlertRuleVOur,
)

CLUSTER = get_cluster()

if CLUSTER == "managed":
    services_to_monitor = [
        (PREVIEWNET_SERVICE, {"route_by": "operator", "urgency": "low"})
    ]
else:
    services_to_monitor = [(MAINNET_SERVICE, {"route_by": "operator"})]


OPERATOR_LABEL_REPLACE_ARGS = (
    '"operator", "$2", "kubernetes_pod_name", "(.*):(.*?)[-/](.*)"'
)

rules = [
    [
        AlertRuleVOur(
            # Each rule must have a unique title
            title=f"[{service.name}][node][operator] "
            "Node connected to less than 80% of other nodes",
            condition="C",
            # Several triggers can be used per alert
            triggers=create_group_by_trigger(
                metric=AlertMetric(
                    expr_template="""
min by (operator) (
    max_over_time(
        label_replace(
            (
                1 + sum by (kubernetes_pod_name)(
                    aptos_connections{{\
{SERVICE_FILTER}, network_id="Validator", role_type="validator"}}
                )
            )
            /
            quantile(0.5, aptos_consensus_current_epoch_validators{{\
{SERVICE_FILTER},  role="validator"}}),
            {OPERATOR_LABEL_REPLACE_ARGS}
        )[{AGGREGATION_INTERVAL}]
    )
)
                    """,
                    name="Percent of connections for a node with fewest "
                    "connections, by a node operator",
                    additional_template_args={
                        "OPERATOR_LABEL_REPLACE_ARGS": OPERATOR_LABEL_REPLACE_ARGS,  # noqa: E501
                    },
                ),
                service=service,
                condition_expression="{SOURCE} < 0.8",
                aggregation_interval="20m",
            ),
            annotations={
                "summary": "Node operator has a node that has fewer than 80% "
                "of connections (over last 20 minutes)",
            },
            labels=labels,
            timeRangeFrom=4 * 3600,
        ),
        AlertRuleVOur(
            # Each rule must have a unique title
            title=f"[{service.name}][node][operator] Node not participating "
            "in consensus",
            condition="C",
            # Several triggers can be used per alert
            triggers=create_group_by_trigger(
                metric=AlertMetric(
                    expr_template="""
sum by (operator) (
    label_replace(
        max by (kubernetes_pod_name) (
            1 - clamp_max(
                sum_over_time(aptos_committed_votes_in_window{{\
{SERVICE_FILTER}, role="validator"}}[{AGGREGATION_INTERVAL}])
                + sum_over_time(aptos_committed_proposals_in_window{{\
{SERVICE_FILTER}, role="validator"}}[{AGGREGATION_INTERVAL}]),
                1
            )
        ),
        {OPERATOR_LABEL_REPLACE_ARGS}
    )
)
                    """,
                    name="Number of nodes not participating in consensus, "
                    "by node operator",
                    additional_template_args={
                        "OPERATOR_LABEL_REPLACE_ARGS": OPERATOR_LABEL_REPLACE_ARGS,  # noqa: E501
                    },
                ),
                service=service,
                condition_expression="{SOURCE} > 0",
                aggregation_interval="20m",
            ),
            annotations={
                "summary": "Number of nodes not participating in consensus, "
                "for a node operator, is at least 1. (over last 20 minutes)",
            },
            labels=labels,
            timeRangeFrom=4 * 3600,
        ),
        AlertRuleVOur(
            # Each rule must have a unique title
            title=f"[{service.name}][node][operator] Broken Validator-VFN "
            "connection",
            condition="C",
            # Several triggers can be used per alert
            triggers=create_group_by_trigger(
                metric=AlertMetric(
                    expr_template="""
sum by (operator) (
    label_replace(
        1 - clamp_max(
            max_over_time(
                aptos_data_client_connected_peers{{\
{SERVICE_FILTER}, role=~"validator_fullnode",peer_type="prioritized_peer"}}\
[{AGGREGATION_INTERVAL}]
            ),
            1
        ),
        {OPERATOR_LABEL_REPLACE_ARGS}
    )
)
                    """,
                    name="Number of VFNs without a connection to the "
                    "validator, by a node operator",
                    additional_template_args={
                        "OPERATOR_LABEL_REPLACE_ARGS": OPERATOR_LABEL_REPLACE_ARGS,  # noqa: E501
                    },
                ),
                service=service,
                condition_expression="{SOURCE} > 0",
                aggregation_interval="20m",
            ),
            annotations={
                "summary": "Number of VFNs without a connection to the "
                "validator, for a node operator, is at least 1. (over last 20m)",
            },
            labels=labels,
            timeRangeFrom=4 * 3600,
        ),
    ]
    for service, labels in services_to_monitor
]

alertgroup = create_alert_group(
    name="operator_node_health",
    # very expensive, and aggregation interval is large,
    # so we can make evaluation interval large too.
    evaluate_interval_s=300,
    evaluate_for="10m",
    # flatten rules
    rules=[rule for rules_for_service in rules for rule in rules_for_service],
)