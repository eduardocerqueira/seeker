#date: 2025-03-24T17:10:41Z
#url: https://api.github.com/gists/ed0fce39120775e9e952cfcad141f018
#owner: https://api.github.com/users/dgarros

from __future__ import annotations

from rich import print as rprint

from infrahub_sdk import InfrahubClient
from infrahub_sdk.async_typer import AsyncTyper
from infrahub_sdk.schema import (
    GenericSchema,
    NodeSchema,
    AttributeSchema,
    AttributeKind,
    SchemaRoot,
    RelationshipSchema,
    RelationshipKind
)
from infrahub_sdk.schema import (
    InfrahubAttributeParam as AttrParam,
)
from infrahub_sdk.schema import (
    InfrahubRelationshipParam as RelParam,
)

app = AsyncTyper()


site = NodeSchema(
    name="Site", 
    namespace="Infra",
    attributes=[
        AttributeSchema(name="name", kind=AttributeKind.TEXT)
    ]
)

device = NodeSchema(
    name="Device",
    namespace="Infra",
    attributes=[
        AttributeSchema(name="name", kind=AttributeKind.TEXT)
    ],
    relationships=[
        RelationshipSchema(name="site", kind=RelationshipKind.ATTRIBUTE),
        RelationshipSchema(name="interfaces", kind=RelationshipKind.COMPONENT)
    ]
)

interface = GenericSchema(
    name="Interface",
    namespace="Infra",
    attributes=[
        AttributeSchema(name="name", kind=AttributeKind.TEXT)
    ],
    relationships=[
        RelationshipSchema(name="device", kind=RelationshipKind.PARENT),
    ]
)

physical_interface = NodeSchema(
    name="PhysicalInterface",
    namespace="Infra",
    inherit_from=[interface.kind],
    attributes=[
        AttributeSchema(name="mtu", kind=AttributeKind.NUMBER),
        AttributeSchema(name="speed", kind=AttributeKind.NUMBER),
        AttributeSchema(name="duplex", kind=AttributeKind.TEXT),
    ]
)

SCHEMA = SchemaRoot(
    version="1.0",
    nodes=[site, device, physical_interface],
    generics=[interface],
)

@app.command()
async def load_schema() -> None:
    client = InfrahubClient()
    rprint(SCHEMA.to_schema_dict())
    response = await client.schema.load(schemas=[SCHEMA.to_schema_dict()], wait_until_converged=True)
    rprint(response)


if __name__ == "__main__":
    app()
