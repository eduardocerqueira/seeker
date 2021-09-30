#date: 2021-09-30T17:03:04Z
#url: https://api.github.com/gists/41e45f2c3241e53601bf3d57cde388f6
#owner: https://api.github.com/users/hoshiyosan

"""
Client used to do actions in vrealize operations.
Use VROPS session to authenticate and perform requests.
"""
import logging
from datetime import datetime
from typing import Dict, List

from flask import Flask

from infrapi.clients.vrops.exceptions import VROPSOperationalError
from infrapi.clients.vrops.session import VROPSSession
from infrapi.clients.vrops.models import VROPSResource, VROPSUserAccount


LOGGER = logging.getLogger(__name__)


class VROPSClient:
    """
    Flask extension used to perform actions on VROPS.
    """

    session: VROPSSession

    def __init__(self, app: Flask = None, config: dict = None):
        self.session = None

        if app or config:
            self.init_app(app, config)

    def init_app(self, app: Flask = None, config: dict = None):
        """
        Configure application either from app config, from given config, or both.
        """
        self.config = {} if app is None else app.config["VROPS_CONFIG"]
        if config:
            self.config.update(config)

        # initialize session
        self.session = VROPSSession(
            username=self.config["username"], password=self.config["password"], base_url=self.config["base_url"]
        )

    def __load_resource(self, resource_data: dict) -> VROPSResource:
        return VROPSResource(
            id=resource_data["identifier"],
            name=resource_data["resourceKey"]["name"],
            created_at=datetime.utcfromtimestamp(resource_data["creationTime"] / 1000),
            adapter_kind=resource_data["resourceKey"]["adapterKindKey"],
            resource_kind=resource_data["resourceKey"]["resourceKindKey"],
            identifiers={
                identifier["identifierType"]["name"]: identifier["value"]
                for identifier in resource_data["resourceKey"]["resourceIdentifiers"]
            },
        )

    def __get_resources(self, adapter_kind: str, resource_kind: str, filters: Dict[str, str] = None) -> List[dict]:
        """
        Query a list of resources from VROPS, with optional filters on resource identifiers.
        :param adapter_kind:  Adapter from which resources are retrieved (e.g. vCloud).
        :param resource_kind: Type of the VROPS resources to retrieve (e.g: ORG)
        :param filters: A dictionary of which keys are the name of resources identifiers and values are expected values for these identifiers.
        """
        if filters is None:
            filters = {}

        response = self.session.get(
            f"/suite-api/api/adapterkinds/{adapter_kind}/resourcekinds/{resource_kind}/resources?",
            params={f"identifiers[{key}]": value for key, value in filters.items()},
        )

        return [self.__load_resource(resource_data) for resource_data in response.json()["resourceList"]]

    def disable_plugin_access(self, org_urn: str):
        """
        Disable vCloud plugin access to VROPS for the given org.
        :param org_urn: unique identifier of the org in vmware
        """
        response = self.session.delete(f"/tenant-app-api/users/{org_urn}")
        if response.status_code != 200:
            raise VROPSOperationalError(f"Failed to disable plugin access for org {org_urn}")

    def enable_plugin_access(self, org_urn: str, org_name: str, password: str):
        """
        Create a user account for an organization in VROPS.
        :param org_urn: unique identifier of the org in vmware
        :param org_name: name of the organization
        :param password: a random string that will serve as password for the account
        """
        # ensure plugin access is disabled before enabling
        self.disable_plugin_access(org_urn)

        # find identifier of the organization in VROPS
        org_resources = self.__get_resources("vCloud", "ORG", filters={
            "UUID": "urn:vcloud:org:9cef9d13-d61e-4db5-9b77-6794f7614d4a"
        })
        try:
            org_resource = org_resources[0]
        except IndexError:
            raise VROPSOperationalError("Org not found in VROPS")

        # enable plugin access by creating a user account for vCloud org's tenants
        response = self.session.post("/tenant-app-api/users", json={
            "username": org_urn,
            "firstName": org_urn,
            "lastName": org_name,
            "password": password,
            "emailAddress": f"{org_urn}@{org_name}.org",
            "enabled": True,
            "groupIds": [],
            "roleNames": ["VCD Tenant Admin"],
            "rolePermissions": [
                {
                    "roleName": "VCD Tenant Admin",
                    "traversalSpecInstances": [
                        {
                            "adapterKind": "vCloud",
                            "resourceKind": "vCloud World",
                            "name": "vCloud Tenant",
                            "selectAllResources": False,
                            "resourceSelections": [
                                {"type": "PROPAGATE", "resourceIds": [org_resource.id]}
                            ],
                        }
                    ],
                    "allowAllObjects": False,
                }
            ],
        })
        if response.status_code != 200:
            raise VROPSOperationalError(f"Something went wrong when enabling user account for org {org_urn}. details: {response.text}")
