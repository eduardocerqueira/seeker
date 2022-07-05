#date: 2022-07-05T17:04:05Z
#url: https://api.github.com/gists/8c14dd2d277126ff45409ed672a813e9
#owner: https://api.github.com/users/cthoyt

"""Update the dependencies."""

import json
from pathlib import Path
from typing import Iterable, Optional

import bioontologies
import bioregistry
import click
from bioontologies.obograph import Graph
from rich import print
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import pystow

OUTPUT_PATH = pystow.join("obo", name="obo_foundry_dependncies.json")
TEST_PREFIXES = {
    "so",
}
SKIP_PREFIXES = {"CHEBI", "NCBITaxon", "PR", "GAZ"}
OBO_URI_PREFIX = "http://purl.obolibrary.org/obo/"
OBO_PREFIXES: set[str] = {
    obo_prefix
    for resource in bioregistry.read_registry().values()
    if (obo_prefix := resource.get_obofoundry_prefix())
}


@click.command()
@click.option("--test", is_flag=True)
@click.option("--no-skip", is_flag=True)
def main(test: bool):
    # test = True
    if OUTPUT_PATH.is_file():
        rv = json.loads(OUTPUT_PATH.read_text())
    else:
        rv = {}
    with logging_redirect_tqdm():
        it = tqdm(
            _prefixes(test=test),
            unit="prefix",
            desc="Gathering OBO dependencies",
        )
        for prefix, obo_prefix in it:
            if prefix in rv:
                tqdm.write(click.style(f"[{prefix}] using cached", fg="green"))
                continue
            it.set_postfix(prefix=obo_prefix)
            try:
                dependencies = lookup_dependencies(prefix)
            except ValueError as e:
                tqdm.write(click.style(f"[{prefix}] error: {e}", fg="red"))
                continue
            if dependencies is None:
                continue
            dependencies = rv[obo_prefix] = sorted(dependencies - {obo_prefix})
            OUTPUT_PATH.write_text(json.dumps(rv, indent=2, sort_keys=True))

            if dependencies:
                dependencies_str = ", ".join(dependencies)
                tqdm.write(f"[{prefix}] depends on {dependencies_str}")
            else:
                tqdm.write(click.style(f"[{prefix}] has no dependencies", fg="yellow"))


def _prefixes(test: bool) -> list[tuple[str, str]]:
    prefixes = sorted(
        (prefix, obo_prefix)
        for prefix, resource in bioregistry.read_registry().items()
        if (obo_prefix := resource.get_obofoundry_prefix())
        and not resource.is_deprecated()
        and obo_prefix not in SKIP_PREFIXES
    )
    if test:
        prefixes = [p for p in prefixes if p[0] in TEST_PREFIXES]
    return prefixes


def lookup_dependencies(prefix: str) -> Optional[set[str]]:
    """Get a set of all dependencies for the ontology."""
    rv: set[str] = set()
    try:
        parse_results = bioontologies.get_obograph_by_prefix(prefix)
    except TypeError as e:
        tqdm.write(click.style(f"[{prefix}] failure: {e}", fg="red"))
        return None
    if not parse_results.graph_document:
        tqdm.write(click.style(f"[{prefix}] could not parse", fg="red"))
        return None
    for graph in parse_results.graph_document.graphs or []:
        rv.update(iter_prefixes(graph=graph))
    return rv.intersection(OBO_PREFIXES)


def iter_prefixes(*, graph: Graph) -> Iterable[str]:
    """Iterate over OBO prefixes used in the graph."""
    for node in tqdm(graph.nodes, leave=False, desc="parsing nodes", unit_scale=True):
        if prefix := get_obo_prefix(node.id):
            yield prefix
    for edge in tqdm(graph.edges, leave=False, desc="parsing edges", unit_scale=True):
        for uri in edge.as_tuple():
            if prefix := get_obo_prefix(uri):
                yield prefix


def get_obo_prefix(uri: str) -> Optional[str]:
    """Parse the OBO prefix from a string, if it's a valid OBO PURL."""
    if not uri.startswith(OBO_URI_PREFIX):
        return None
    return uri.removeprefix(OBO_URI_PREFIX).rsplit("_", maxsplit=1)[0]


if __name__ == "__main__":
    main()
