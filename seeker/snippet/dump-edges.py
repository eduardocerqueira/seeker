#date: 2025-02-26T17:04:23Z
#url: https://api.github.com/gists/a28d037364df97bc218aee72c7f6f6b3
#owner: https://api.github.com/users/nick-barrett

from dataclasses import dataclass
import json
import asyncio
import os
from typing import Any, AsyncGenerator
import aiohttp


def read_env(name: str) -> str:
    value = os.getenv(name)
    assert value is not None, f"missing environment var {name}"
    return value


@dataclass
class CommonData:
    vco: str
    token: "**********"
    enterprise_id: int
    session: aiohttp.ClientSession

    def __post_init__(self):
        self.validate()

        self.session.headers.update({"Authorization": "**********"

    def validate(self):
        if any(
            missing_inputs := [
                v is None for v in [self.vco, self.token, self.enterprise_id]
            ]
        ):
            raise ValueError(f"missing input data: {missing_inputs}")


async def do_portal(c: CommonData, method: str, params: dict):
    async with c.session.post(
        f"https://{c.vco}/portal/",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        },
    ) as req:
        resp = await req.json()
        if "result" not in resp:
            raise ValueError(json.dumps(resp, indent=2))
        return resp["result"]


async def get_enterprise_edge_list_raw(
    c: CommonData,
    with_params: list[str] | None,
    filters: dict | None,
    next_page: str | None = None,
) -> dict[str, list | dict]:
    params_object: dict[str, Any] = {
        "enterpriseId": c.enterprise_id,
        "limit": 500,
        "sortBy": [{"attribute": "edgeState", "type": "ASC"}],
    }

    if with_params:
        params_object["with"] = with_params

    if filters:
        params_object["filters"] = filters
    else:
        params_object["_filterSpec"] = True

    if next_page:
        params_object["nextPageLink"] = next_page

    return await do_portal(c, "enterprise/getEnterpriseEdges", params_object)


async def get_enterprise_edge_list_full_dict(
    c: CommonData, with_params: list[str] | None, filters: dict | None
) -> AsyncGenerator[dict[Any, Any], None]:
    next_page = None
    more = True

    while more:
        resp = await get_enterprise_edge_list_raw(c, with_params, filters, next_page)

        meta = resp.get("metaData", {})
        more = meta.get("more", False)
        next_page = meta.get("nextPageLink", None)

        data = resp.get("data", [])
        for d in data:
            yield d


async def main(session: aiohttp.ClientSession):
    data = CommonData(
        read_env("VCO"), read_env("VCO_TOKEN"), int(read_env("ENT_ID")), session
    )

    edges_full: list[dict[str, Any]] = []

    async for edge in get_enterprise_edge_list_full_dict(
        data,
        [
            "ha",
            "configuration",
            "licenses",
            "certificateSummary",
            "analyticsMode",
            "isAssignedGatewayQuiesced",
            "selfHealing",
            "secureDeviceSecrets",
            "site",
            "recentLinks",
            "cloudServices",
            "nvsFromEdge",
            "vnfs",
        ],
        None,
    ):
        edges_full.append(edge)

    with open("edge-export.json", "w") as f:
        json.dump(edges_full, f, indent=2)


async def main_wrapper():
    async with aiohttp.ClientSession() as session:
        await main(session)


if __name__ == "__main__":
    asyncio.run(main_wrapper())
)
