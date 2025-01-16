#date: 2025-01-16T16:59:40Z
#url: https://api.github.com/gists/75e49b2f183263cd4f57b7c66c6a25a9
#owner: https://api.github.com/users/cnolanminich

import json
import textwrap
from typing import Any, Mapping, List, Tuple
from dagster import (
    AutomationCondition,
    AssetKey,
    BackfillPolicy,
    DailyPartitionsDefinition,
    job,
    op,
    AssetExecutionContext,
    WeeklyPartitionsDefinition,
    load_assets_from_package_module,
    AssetSpec
)
from dagster_cloud.dagster_insights import dbt_with_snowflake_insights
from dagster_dbt import (
    DbtCliResource,
    DagsterDbtTranslator,
    default_metadata_from_dbt_resource_props,
    DagsterDbtTranslatorSettings,
)
from dagster_dbt.asset_decorator import dbt_assets
from dagster_dbt.freshness_builder import build_freshness_checks_from_dbt_assets
from dagster import build_sensor_for_freshness_checks
# assumes all assets are housed in a file called assets.py
from . import assets
# assumes you have a dbt_project object
from .resources import dbt_project

# creates a set of Dagster asset specs (could be source assets but I think this might be more flexible for you)
def create_external_assets_from_dbt_sources(manifest_path: str) -> List[Tuple[str, str]]:
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Extract all sources from nodes
    all_sources = []
    for node in manifest['nodes'].values():
        if 'sources' in node:
            all_sources.extend(node['sources'])
    
    # Convert to set of tuples to get unique values
    unique_sources = list(set((source[0].upper(), source[1]) for source in all_sources))
    
    # example if you had assets in different files
    #assets = raw_data_assets + forecasting_assets
    # add group name and key prefixes to match what's in defintions.py
    assets = load_assets_from_package_module(assets, group_name="FORECASTING", key_prefix="FORECASTING")
    asset_details = []
    for asset_def in assets:
            for asset_spec in asset_def.specs:
                # Append the check name and asset key to the list
                asset_details.append((asset_spec.key.path))
    # Convert asset_details to set of tuples for comparison
    #existing_assets = {(asset[0].upper(), asset[1]) for asset in asset_details}
    # you might be able to ignore this if statement -- I had some assets with no key prefixes and some with for testing purposes
    existing_assets = {(asset[0],) if len(asset) == 1 else (asset[0].upper(), asset[1]) for asset in asset_details}
    # Find sources that don't exist in asset_details
    new_sources = [AssetSpec(source) for source in unique_sources if source not in existing_assets and source[1] != "predicted_orders"]
    return new_sources

# Create external assets
external_assets = create_external_assets_from_dbt_sources(dbt_project.manifest_path)