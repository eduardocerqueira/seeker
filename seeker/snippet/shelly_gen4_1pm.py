#date: 2025-06-24T17:11:09Z
#url: https://api.github.com/gists/6cb69bc3a32a727c4b06b3dd4210c851
#owner: https://api.github.com/users/tube0013

"""Shelly V2 Quirk using the modern, multi-file builder pattern.

This file demonstrates how the quirk would be structured across two files:
1. zhaquirks/shelly/__init__.py (for shared, manufacturer-specific code)
2. zhaquirks/shelly/s4sw001p8eu.py (for device-specific entity definitions)
"""

from __future__ import annotations

import enum
import logging

from zigpy.zcl.clusters.manufacturer_specific import ManufacturerSpecificCluster

from zhaquirks.v2 import (
    EntityCategory,
    Number,
    QuirkBuilder,
    Select,
    Sensor,
    Switch,
    Text,
)

_LOGGER = logging.getLogger(__name__)

# #############################################################################
#
#   Part 1: Shared Manufacturer Logic
#   (This code would live in `zhaquirks/shelly/__init__.py`)
#
# #############################################################################

SHELLY_MANUFACTURER = "Shelly"
SHELLY_MANUFACTURER_ID = 0x1490
RPC_CLUSTER_ID = 0xFC01
WIFI_SETUP_CLUSTER_ID = 0xFC02
SHELLY_EP = 239


class WifiAction(enum.Enum):
    """Actions for the WiFi Setup cluster."""

    NOP = 0
    APPLY = 1
    RESET = 2


class ShellyRPCCluster(ManufacturerSpecificCluster):
    """Shelly RPC cluster (0xFC01)."""

    cluster_id = RPC_CLUSTER_ID
    name = "Shelly RPC"
    ep_attribute = "shelly_rpc"
    manufacturer_id_override = SHELLY_MANUFACTURER_ID

    attributes = ManufacturerSpecificCluster.attributes.copy()
    attributes.update(
        {
            0x0000: ("data", bytes),
            0x0001: ("tx_ctl", "uint16_t"),
            0x0002: ("rx_ctl", "uint16_t"),
        }
    )


class ShellyWiFiSetupCluster(ManufacturerSpecificCluster):
    """Shelly WiFi Setup cluster (0xFC02)."""

    cluster_id = WIFI_SETUP_CLUSTER_ID
    name = "Shelly WiFi Setup"
    ep_attribute = "shelly_wifi_setup"
    manufacturer_id_override = SHELLY_MANUFACTURER_ID

    attributes = ManufacturerSpecificCluster.attributes.copy()
    attributes.update(
        {
            0x0000: ("status", str),
            0x0001: ("ip", str),
            0x0002: ("action", "uint8_t"),
            0x0003: ("dhcp", bool),
            0x0004: ("enable", bool),
            0x0005: ("ssid", str),
            0x0006: "**********"
        }
    )


# #############################################################################
#
#   Part 2: Device-Specific Quirk Definition
#   (This code would live in `zhaquirks/shelly/s4sw001p8eu.py`)
#
# #############################################################################

# In a real multi-file setup, the following imports would be used:
# from . import (
#     SHELLY_EP,
#     SHELLY_MANUFACTURER,
#     WIFI_SETUP_CLUSTER_ID,
#     RPC_CLUSTER_ID,
#     ShellyRPCCluster,
#     ShellyWiFiSetupCluster,
#     WifiAction,
# )

# Define the entities for this specific device
SHELLY_1PM_G4_ENTITIES = [
    # --- WiFi Setup Entities ---
    Sensor(
        "WiFi Status",
        "status",
        WIFI_SETUP_CLUSTER_ID,
        endpoint_id=SHELLY_EP,
        icon="mdi:wifi",
    ),
    Sensor(
        "WiFi IP", "ip", WIFI_SETUP_CLUSTER_ID, endpoint_id=SHELLY_EP, icon="mdi:ip"
    ),
    Switch(
        "WiFi Enabled",
        "enable",
        WIFI_SETUP_CLUSTER_ID,
        endpoint_id=SHELLY_EP,
        entity_category=EntityCategory.CONFIG,
    ),
    Switch(
        "WiFi DHCP",
        "dhcp",
        WIFI_SETUP_CLUSTER_ID,
        endpoint_id=SHELLY_EP,
        entity_category=EntityCategory.CONFIG,
    ),
    Text(
        "WiFi SSID",
        "ssid",
        WIFI_SETUP_CLUSTER_ID,
        endpoint_id=SHELLY_EP,
        entity_category=EntityCategory.CONFIG,
    ),
    Text(
        "WiFi Password",
        "password",
        WIFI_SETUP_CLUSTER_ID,
        endpoint_id=SHELLY_EP,
        write_only=True,
        entity_category=EntityCategory.CONFIG,
    ),
    Select(
        "WiFi Action",
        "action",
        WIFI_SETUP_CLUSTER_ID,
        WifiAction,
        endpoint_id=SHELLY_EP,
        entity_category=EntityCategory.CONFIG,
    ),
    # --- RPC Entities ---
    Number(
        "RPC Tx Control",
        "tx_ctl",
        RPC_CLUSTER_ID,
        endpoint_id=SHELLY_EP,
        entity_category=EntityCategory.CONFIG,
    ),
    Sensor(
        "RPC Rx Control",
        "rx_ctl",
        RPC_CLUSTER_ID,
        endpoint_id=SHELLY_EP,
    ),
    Text(
        "RPC Data",
        "data",
        RPC_CLUSTER_ID,
        endpoint_id=SHELLY_EP,
        entity_category=EntityCategory.CONFIG,
    ),
]


# Use the QuirkBuilder to create and register the quirk
QuirkBuilder(
    SHELLY_MANUFACTURER, "S4SW-001P8EU"
).replace_cluster_occurrences(
    [ShellyRPCCluster, ShellyWiFiSetupCluster]
).declare_entities(
    SHELLY_1PM_G4_ENTITIES
).add_to_registry()

ry()

