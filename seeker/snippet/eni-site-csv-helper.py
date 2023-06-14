#date: 2023-06-14T16:56:18Z
#url: https://api.github.com/gists/9282dccf8d95722996568c8b0d485126
#owner: https://api.github.com/users/nbarrettvmw

"""

Description: Read analytics-enabled edges from the VCO and maintain set of sites in ENI.
Author: Nick Barrett <nbarrett@vmware.com>

Requirements:
- python 3.11 - not tested below this version

External dependencies:
- python-dotenv https://pypi.org/project/python-dotenv/
- dataclass-csv https://pypi.org/project/dataclass-csv/

Environment or .env variables:
- VCO => FQDN of customer VCO
- VCO_TOKEN = "**********"
- ENT_ID => Enterprise ID for the customer in the VCO (integer, not logical ID)

Usage: 
python3 eni-site-csv-helper.py --env-file .my.env --old-sites eni-sites-old.csv --output-sites eni-sites-new.csv

"""

import json
import os
import csv
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass, field
from requests import Session, session
from typing import cast

from dataclass_csv import DataclassReader, DataclassWriter
import dotenv


@dataclass
class CommonData:
    vco: str
    token: "**********"
    enterprise_id: int
    session: Session = field(init=False)

    def __post_init__(self):
        self.validate()

        s = session()
        s.headers.update({"Authorization": "**********"

        self.session = s

    def validate(self):
        if any(
            missing_inputs := [
                v is None for v in [self.vco, self.token, self.enterprise_id]
            ]
        ):
            raise ValueError(f"missing input data: {missing_inputs}")


def do_portal(shared: CommonData, method: str, params: dict):
    resp = shared.session.post(
        f"https://{shared.vco}/portal/",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        },
    ).json()
    if "result" not in resp:
        raise ValueError(json.dumps(resp, indent=2))
    return resp["result"]


@dataclass
class EdgeInfoRecord:
    name: str
    crawler_ids: str = ""
    edge_ids: str = ""
    controller_ips: str = ""
    subnets: str = ""
    aps: str = ""
    lat: float = 1000.0
    lon: float = 1000.0
    place_name: str = ""
    ignore: str = "false"


def is_edge_offline_too_long(last_contact: str | None) -> bool:
    """2023-06-14T16:33:46.635Z"""
    try:
        if last_contact is not None:
            dt = datetime.fromisoformat(last_contact)
            delta = datetime.now(timezone.utc) - dt
            delta_seconds = delta.total_seconds()

            res = delta_seconds > 60 * 60 * 24 * 3

            return res
        else:
            return True
    except Exception as e:
        print(f"exception checking last contact on edge: {e}")
        return True


def read_new_analytics_edges(
    shared: CommonData, completed_sites: set[str]
) -> list[EdgeInfoRecord]:
    raw_edges: list[dict] = do_portal(
        shared,
        "enterprise/getEnterpriseEdgeList",
        {"enterpriseId": shared.enterprise_id, "with": ["site", "analyticsMode"]},
    )

    print(f"{len(raw_edges)} total edges loaded from VCO")

    resp = []
    for raw_edge in raw_edges:
        analytics_mode = raw_edge.get("analyticsMode", None)
        activation_state = raw_edge.get("activationState", None)
        last_contact = cast(str | None, raw_edge.get("lastContact", None))

        if (
            analytics_mode == "SDWAN_ANALYTICS"
            and activation_state == "ACTIVATED"
            and not is_edge_offline_too_long(last_contact)
        ):
            edge_name = raw_edge.get("name", "")
            edge_logical_id = raw_edge.get("logicalId", "")
            site = raw_edge.get("site", {})
            # validated with isinstance below
            lat = cast(float | None, site.get("lat", None))
            lon = cast(float | None, site.get("lon", None))
            # make sure we don't create any sites without name, logId, or proper locations
            # also don't create a site that we've already done previously
            if (
                len(edge_name) > 1
                and len(edge_logical_id) > 4
                and isinstance(lat, float)
                and isinstance(lon, float)
                and edge_logical_id not in completed_sites
            ):
                resp.append(
                    EdgeInfoRecord(edge_name, "", edge_logical_id, "", "", "", lat, lon)
                )

    return resp


def read_existing_sites(file_path: str | None) -> set[str]:
    result = set()

    if file_path:
        with open(file_path, "r") as f:
            r = DataclassReader(f, EdgeInfoRecord, quoting=csv.QUOTE_ALL)
            r.map("Name").to("name")
            r.map("CrawlerIds").to("crawler_ids")
            r.map("EdgeIds").to("edge_ids")
            r.map("ControllerIps").to("controller_ips")
            r.map("Subnets").to("subnets")
            r.map("APs").to("aps")
            r.map("Lat").to("lat")
            r.map("Lng").to("lon")
            r.map("Place name").to("place_name")
            r.map("Ignore?").to("ignore")
            for row in r:
                result.add(row.edge_ids)

    print(f"{len(result)} existing sites loaded from CSV")

    return result


def readenv(name: str) -> str:
    val = os.getenv(name)
    assert val is not None, f"missing env var {name}"
    return val


def main(env_file: str | None, input_csv_path: str | None, output_csv_path: str):
    if env_file:
        dotenv.load_dotenv(env_file)

    shared = "**********"

    completed_sites = read_existing_sites(input_csv_path)
    new_edges = read_new_analytics_edges(shared, completed_sites)

    if len(new_edges) > 0:
        print(
            f"{len(new_edges)} new sites to be added. writing as CSV to {output_csv_path}..."
        )

        with open(output_csv_path, "w", newline="") as csv_file:
            writer = DataclassWriter(
                csv_file, new_edges, EdgeInfoRecord, quoting=csv.QUOTE_ALL
            )
            writer.map("name").to("Name")
            writer.map("crawler_ids").to("CrawlerIds")
            writer.map("edge_ids").to("EdgeIds")
            writer.map("controller_ips").to("ControllerIps")
            writer.map("subnets").to("Subnets")
            writer.map("aps").to("APs")
            writer.map("lat").to("Lat")
            writer.map("lon").to("Lng")
            writer.map("place_name").to("Place name")
            writer.map("ignore").to("Ignore?")
            writer.write()
    else:
        print("no new sites to add")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, help="Environment variable filename")
    parser.add_argument(
        "-i",
        "--input-csv-filename",
        type=str,
        dest="input_csv",
        help="Filename of a previous run's CSV output",
    )
    parser.add_argument(
        "-o",
        "--output-csv-filename",
        type=str,
        dest="output_csv",
        help="Filename to save this run's CSV output to",
        required=True,
    )

    args = parser.parse_args()
    main(args.env, args.input_csv, args.output_csv)
parser.parse_args()
    main(args.env, args.input_csv, args.output_csv)
