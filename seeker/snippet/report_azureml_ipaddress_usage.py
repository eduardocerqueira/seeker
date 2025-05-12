#date: 2025-05-12T16:59:53Z
#url: https://api.github.com/gists/c12d5bafc563096e00a44ed267152e04
#owner: https://api.github.com/users/RH-TLagrone

#!/usr/bin/env python
# /// script
# requires-python = ">=3.12,<4"
# dependencies = [
#   "azure-identity",
#   "azure-mgmt-resource",
#   "azure-mgmt-network",
#   "azure-ai-ml",
# ]
# ///
import dataclasses
import json
import os
import re
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from ipaddress import ip_network
from typing import NamedTuple, cast, override

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient

COMPUTE_URI_FORMAT: str = (
    r"/subscriptions/{subscription_id}"
    r"/resourceGroups/{resource_group_name}"
    r"/providers/Microsoft.MachineLearningServices"
    r"/workspaces/{workspace_name}"
    r"/computes/{compute_name}"
)
SUBNET_URI_FORMAT: str = (
    r"/subscriptions/{subscription_id}"
    r"/resourceGroups/{resource_group_name}"
    r"/providers/Microsoft.Network"
    r"/virtualNetworks/{virtual_network_name}"
    r"/subnets/{subnet_name}"
)
WORKSPACE_URI_FORMAT: str = (
    r"/subscriptions/{subscription_id}"
    r"/resourceGroups/{resource_group_name}"
    r"/providers/Microsoft.MachineLearningServices"
    r"/workspaces/{workspace_name}"
)

COMPUTE_URI_PATTERN: re.Pattern[str] = re.compile(
    (
        r"/subscriptions/(?P<subscription_id>[^/]+)"
        r"/resourceGroups/(?P<resource_group_name>[^/]+)"
        r"/providers/Microsoft\.MachineLearningServices"
        r"/workspaces/(?P<workspace_name>[^/]+)"
        r"/computes/(?P<compute_name>[^/]+)"
    )
)
SUBNET_URI_PATTERN: re.Pattern[str] = re.compile(
    (
        r"/subscriptions/(?P<subscription_id>[^/]+)"
        r"/resourceGroups/(?P<resource_group_name>[^/]+)"
        r"/providers/Microsoft\.Network"
        r"/virtualNetworks/(?P<virtual_network_name>[^/]+)"
        r"/subnets/(?P<subnet_name>[^/]+)"
    )
)
WORKSPACE_URI_PATTERN: re.Pattern[str] = re.compile(
    (
        r"/subscriptions/(?P<subscription_id>[^/]+)"
        r"/resourceGroups/(?P<resource_group_name>[^/]+)"
        r"/providers/Microsoft\.MachineLearningServices"
        r"/workspaces/(?P<workspace_name>[^/]+)"
    )
)


class ComputeUri(NamedTuple):
    subscription_id: str
    resource_group_name: str
    workspace_name: str
    compute_name: str

    @override
    def __str__(self) -> str:
        return COMPUTE_URI_FORMAT.format(**self._asdict())

    @classmethod
    def parse(cls, uri: str) -> "ComputeUri":
        if match := COMPUTE_URI_PATTERN.fullmatch(uri):
            return ComputeUri(*match.groups())
        else:
            raise ValueError(f"Not an Azure Machine Learning Compute resource id: {uri}")  # fmt: off


class SubnetUri(NamedTuple):
    subscription_id: str
    resource_group_name: str
    virtual_network_name: str
    subnet_name: str

    @override
    def __str__(self) -> str:
        return SUBNET_URI_FORMAT.format(**self._asdict())

    @classmethod
    def parse(cls, uri: str) -> "SubnetUri":
        if match := SUBNET_URI_PATTERN.fullmatch(uri):
            return SubnetUri(*match.groups())
        else:
            raise ValueError(f"Not an Azure Subnet resource id: {uri}")  # fmt: off


class WorkspaceUri(NamedTuple):
    subscription_id: str
    resource_group_name: str
    workspace_name: str

    @override
    def __str__(self) -> str:
        return WORKSPACE_URI_FORMAT.format(**self._asdict())

    @classmethod
    def parse(cls, uri: str) -> "WorkspaceUri":
        if match := WORKSPACE_URI_PATTERN.fullmatch(uri):
            return WorkspaceUri(*match.groups())
        else:
            raise ValueError(f"Not an Azure Machine Learning Workspace resource id: {uri}")  # fmt: off


@dataclass(order=True, frozen=True, slots=True)
class ComputeInfo:
    subscription_id: str
    resource_group_name: str
    workspace_name: str
    name: str
    type: str = field(compare=False)
    nodes: int = field(compare=False)
    virtual_network_name: str = field(compare=False)
    subnet_name: str = field(compare=False)

    subnet_id: InitVar[str] = field(compare=False)
    _uri: ComputeUri = field(init=False, compare=False)

    @property
    def uri(self) -> ComputeUri:
        return self._uri

    def __post_init__(self):
        self._uri = ComputeUri(
            self.subscription_id,
            self.resource_group_name,
            self.workspace_name,
            self.name,
        )


@dataclass(order=True, frozen=True, slots=True)
class SubnetInfo:
    subscription_id: str
    resource_group_name: str
    virtual_network_name: str
    name: str
    total_raw_addresses: int = field(compare=False)
    total_usable_addresses: int = field(compare=False)
    total_addresses_used: int = field(default=0, compare=False)

    total_raw_addresses_remaining: int = field(init=False, compare=False)
    total_usable_addresses_remaining: int = field(init=False, compare=False)

    _uri: SubnetUri = field(init=False, compare=False)

    @property
    def uri(self) -> SubnetUri:
        return self._uri

    def __post_init__(self):
        self._uri = SubnetUri(
            self.subscription_id,
            self.resource_group_name,
            self.virtual_network_name,
            self.name,
        )
        self.total_raw_addresses_remaining = self.total_raw_addresses - self.total_addresses_used  # fmt: off
        self.total_usable_addresses_remaining = self.total_usable_addresses - self.total_addresses_used  # fmt: off


def describe_computes(client: MLClient) -> list[ComputeInfo]:
    infos = []
    for compute in client.compute.list():
        match compute.type:
            case "amlcompute":
                nodes = compute.max_instances
            case "computeinstance":
                nodes = 1
            case "kubernetes":
                continue
            case _:
                raise ValueError(f"Unrecognized compute type: {compute.type}")
        info = ComputeInfo(
            client.subscription_id,
            client.resource_group_name,
            client.workspace_name,
            compute.name,
            compute.type,
            nodes,
            compute.network_settings.vnet_name,
            subnet_name=compute.network_settings.subnet.split("/")[-1],
            subnet_id=compute.network_settings.subnet,
        )
        infos.append(info)
    return infos


def describe_subnet(client: NetworkManagementClient, uri: SubnetUri) -> SubnetInfo:
    subnet = client.subnets.get(uri.resource_group_name, uri.virtual_network_name, uri.subnet_name)  # fmt: off
    networks = [ip_network(cidr) for cidr in subnet.address_prefixes]
    return SubnetInfo(
        uri.subscription_id,
        uri.resource_group_name,
        uri.virtual_network_name,
        uri.subnet_name,
        total_raw_addresses=sum(nw.num_addresses for nw in networks),
        total_usable_addresses=sum(max(0, nw.num_addresses - 2) for nw in networks),
    )


def main():
    sub_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    cred = DefaultAzureCredential()

    rm = ResourceManagementClient(cred, sub_id)
    computes: list[ComputeInfo] = []
    for ws in rm.resources.list(filter="resourceType eq 'Microsoft.MachineLearningServices/workspaces'"):  # fmt: off
        ws_uri = WorkspaceUri.parse(ws.id)
        ml = MLClient(cred, ws_uri.subscription_id, ws_uri.resource_group_name, ws_uri.workspace_name)  # fmt: off
        computes.extend(describe_computes(ml))

    nm = NetworkManagementClient(cred, sub_id)
    subnets: list[SubnetInfo] = []
    for sn_uri in sorted(set(cp.uri for cp in computes)):
        subnets.append(describe_subnet(nm, cast(SubnetUri, sn_uri)))

    total_addresses_used_by_subnet: dict[SubnetUri, int] = defaultdict(lambda: 0)
    for compute in computes:
        sn_uri = SubnetUri.parse(compute.subnet_id)
        total_addresses_used_by_subnet[sn_uri] += compute.nodes
    subnets = [
        dataclasses.replace(
            sn, total_addresses_used=total_addresses_used_by_subnet[sn.uri]
        )
        for sn in subnets
    ]

    data = {
        "computes": list(map(dataclasses.asdict, computes)),
        "subnets": list(map(dataclasses.asdict, subnets)),
    }
    text = json.dumps(data)
    print(text)


if __name__ == "__main__":
    main()
