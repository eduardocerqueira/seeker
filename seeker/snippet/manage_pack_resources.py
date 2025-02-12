#date: 2025-02-12T16:59:19Z
#url: https://api.github.com/gists/f8d550f7fec40b5ae658b321881f3cd9
#owner: https://api.github.com/users/cognifloyd

import os
import pathlib
import time
from collections import defaultdict
from typing import DefaultDict, Dict, TYPE_CHECKING, Union, Optional
from urllib.parse import urlparse

# noinspection PyPackageRequirements
import yaml

# noinspection PyPackageRequirements
from st2client.commands.action import (
    LIVEACTION_STATUS_CANCELING,
    LIVEACTION_STATUS_ABANDONED,
    LIVEACTION_STATUS_CANCELED,
    LIVEACTION_STATUS_FAILED,
    LIVEACTION_STATUS_TIMED_OUT,
    LIVEACTION_STATUS_SUCCEEDED,
    LIVEACTION_STATUS_RUNNING,
    LIVEACTION_STATUS_DELAYED,
    LIVEACTION_STATUS_PAUSED,
    LIVEACTION_STATUS_PAUSING,
    LIVEACTION_STATUS_SCHEDULED,
    LIVEACTION_STATUS_REQUESTED,
)

# noinspection PyPackageRequirements
from st2client.models import LiveAction

# noinspection PyPackageRequirements
from urllib3 import disable_warnings

if TYPE_CHECKING:
    from python_runner.python_action_wrapper import ActionService

# noinspection PyPackageRequirements
from st2client.client import Client
from st2common.config import cfg
from st2common.runners.base_action import Action

# These are known resources that we can enable/disable
# key is resource_type in pack_resources.yaml
# value is resource_type used in st2client.client.Client.managers
RESOURCE_TYPES_MAP = {
    "rules": "Rule",
    "policies": "Policy",
    "sensors": "Sensor",
    "triggers": "Trigger",
    "actions": "Action",
    "aliases": "ActionAlias",
}

# Some constants helpful to execute st2 actions
ST2_TERMINAL_STATUSES = [
    LIVEACTION_STATUS_CANCELING,
    LIVEACTION_STATUS_ABANDONED,
    LIVEACTION_STATUS_CANCELED,
    LIVEACTION_STATUS_FAILED,
    LIVEACTION_STATUS_TIMED_OUT,
]

ST2_SUCCESSFUL_STATUSES = [LIVEACTION_STATUS_SUCCEEDED]

ST2_RUNNING_STATUSES = [
    LIVEACTION_STATUS_RUNNING,
    LIVEACTION_STATUS_DELAYED,
    LIVEACTION_STATUS_PAUSED,
    LIVEACTION_STATUS_PAUSING,
    LIVEACTION_STATUS_SCHEDULED,
]

PENDING_STATUSES = [
    LIVEACTION_STATUS_REQUESTED,
    LIVEACTION_STATUS_SCHEDULED,
    LIVEACTION_STATUS_RUNNING,
    LIVEACTION_STATUS_CANCELING,
]

RUNNING_STATUSES = [LIVEACTION_STATUS_SCHEDULED, LIVEACTION_STATUS_RUNNING]

# Max execution time per action by default (in seconds)
MAX_DEFAULT_TIME_WAIT = 300
# Max time to wait after kicking off action execution (in seconds)
MAX_ACTION_KICK_OFF_TIME = 10
# Poll interval (in seconds)
ACTION_EXECUTION_POLL_INTERVAL = 2


def make_fake_result(result: dict, final_res: bool) -> dict:
    # this is to imitate success or failure for executed actions when final success is
    # determined later
    if final_res:
        result["want_enabled"] = result["enabled_after"] = True
    else:
        result["want_enabled"] = True
        result["enabled_after"] = False
    return result


class ManagePackResources(Action):
    if TYPE_CHECKING:
        action_service: ActionService

    def __init__(self, config=None, action_service=None):
        super().__init__(config, action_service)
        if not config.get("local_test_with"):
            try:
                webui_base_url = urlparse(cfg.CONF.webui.webui_base_url)
                # noinspection PyTypeChecker
                self.webui_base_domain: str = webui_base_url.hostname
            except Exception as exc:
                self.logger.warning(
                    "Could not parse cfg.CONF.webui.webui_base_url: {}".format(exc)
                )
                try:
                    webui_base_url = urlparse(os.environ.get("ST2_API_URL"))
                    # noinspection PyTypeChecker
                    self.webui_base_domain: str = webui_base_url.hostname
                except Exception as exc:
                    self.logger.warning(
                        "Could not parse env var ST2_API_URL: {}".format(exc)
                    )
                    raise exc
        else:
            self.webui_base_domain: str = config.get("local_test_with")
            disable_warnings()

        self.client = Client()
        self.check_mode = False

    def run(self, from_packs: list = None, check_mode=False):
        self.check_mode = check_mode

        if len(from_packs) == 1:
            packs = {from_packs[0]: self.client.packs.get_by_ref_or_id(from_packs[0])}
        else:
            packs = {p.ref: p for p in self.client.packs.get_all()}

        results = {}
        for pack_name in from_packs:
            pack = packs[pack_name]
            results[pack_name] = self.resources_in_pack(pack_name, pack)

        success = all(
            result.get("want_enabled", True) == result.get("enabled_after", False)
            for pack_results in results.values()
            for resource_results in pack_results.values()
            for result in resource_results.values()
        )

        return {"success": success, "packs": results}

    def resources_in_pack(
        self, pack_name, pack
    ) -> Dict[str, Dict[str, Dict[str, Union[bool, str]]]]:

        # {resource_type: {resource_name:
        #   {want_enabled: bool, enabled_before: bool, enabled_after: bool, error_message: str}
        # }}
        results: DefaultDict[str, Dict[str, Dict[str, Union[bool, str]]]] = defaultdict(
            dict
        )

        pack_path = pathlib.Path(pack.path)
        resources_file_path = pack_path / "pack_resources.yaml"
        if not resources_file_path.exists():
            return results

        with resources_file_path.open("r") as resources_file:
            all_resources = yaml.safe_load(resources_file)
        resources = all_resources.get(self.webui_base_domain, {})
        action_execution_index = 0
        for resource_type, resource_list in resources.items():
            for resource_name in resource_list:
                client_resource_type = RESOURCE_TYPES_MAP.get(resource_type)
                if not client_resource_type:
                    # ignore unknown resource_type
                    continue

                if client_resource_type == "Action" and isinstance(resource_name, dict):
                    action_def = dict(resource_name)
                    resource_name = action_def["ref"]
                else:
                    action_def = None
                # we support <pack>.<resource> and just <resource>
                pack_prefix = f"{pack_name}."
                pack_prefix_len = len(pack_prefix)
                enabled = not resource_name.startswith("^")
                resource_name = resource_name.lstrip("^")
                name = (
                    resource_name[pack_prefix_len:]
                    if resource_name.startswith(pack_prefix)
                    else resource_name
                )

                if action_def is None:
                    result = self.manage_resource(
                        resource_type=client_resource_type,
                        name=name,
                        pack=pack_name,
                        enabled=enabled,
                    )
                else:
                    result = self.execute_action(
                        action_def=action_def,
                    )
                    action_execution_index += 1
                    # since we can technically run multiple actions with the same ref,
                    # make keys unique
                    resource_name = "{}-{}".format(
                        action_execution_index, resource_name
                    )
                results[resource_type][resource_name] = result
        return results

    def execute_action(self, action_def: dict) -> Dict[str, Union[bool, str]]:
        execution_instance = LiveAction()
        execution_instance.action = action_def.get("ref")
        if action_def.get("input"):
            execution_instance.parameters = action_def.get("input")
        try:
            #  verify response to the client and make sure that the requested action exists and
            #  was accepted for execution
            execution = self.client.liveactions.create(execution_instance)
        except Exception as exc:
            result = {
                "success": False,
                "error_message": "{}".format(exc),
            }
            return make_fake_result(result, False)
        self.logger.debug(
            "Created action execution {} for: {}".format(execution.id, action_def)
        )
        if action_def.get("wait", False):
            # wait for the execution to finish or timeout
            max_time_wait = action_def.get("timeout", MAX_DEFAULT_TIME_WAIT)
            start_time = int(time.time())
            now_time = start_time
            while (
                execution.status in PENDING_STATUSES
                and now_time - start_time < max_time_wait
            ):
                time.sleep(ACTION_EXECUTION_POLL_INTERVAL)
                execution = self.client.liveactions.get_by_id(execution.id)
                self.logger.debug(
                    "Action execution {} status: {}".format(
                        execution.id, execution.status
                    )
                )
                now_time = int(time.time())

            if execution.status in PENDING_STATUSES:
                # if timed out, cancel it and ignore response
                self.client.executions.delete_by_id(instance_id=execution.id)
                result = {
                    "success": False,
                    "error_message": "Action execution {} (execution id: "
                    "{}) timed out.".format(execution_instance.action, execution.id),
                }
            elif execution.status in ST2_TERMINAL_STATUSES:
                result = {
                    "success": False,
                    "error_message": "Action execution {} (execution id: "
                    "{}) failed.".format(execution_instance.action, execution.id),
                }
            else:
                result = {
                    "success": True,
                    "error_message": "Action execution {} (execution id: {}) kicked off "
                    "and succeeded.".format(execution_instance.action, execution.id),
                }
        else:
            # wait for the action to be requested/scheduled
            start_time = int(time.time())
            now_time = start_time
            while (
                execution.status not in RUNNING_STATUSES
                and execution.status not in ST2_SUCCESSFUL_STATUSES
            ) and now_time - start_time < MAX_ACTION_KICK_OFF_TIME:
                time.sleep(ACTION_EXECUTION_POLL_INTERVAL)
                execution = self.client.liveactions.get_by_id(execution.id)
                now_time = int(time.time())
                if execution.status in ST2_TERMINAL_STATUSES:
                    result = {
                        "success": False,
                        "want_enabled": True,
                        "enabled_after": False,
                        "error_message": "Action execution {} (execution id: "
                        "{}) failed with status: "
                        "{}.".format(
                            execution_instance.action, execution.id, execution.status
                        ),
                    }
                    return make_fake_result(result, False)
            result = {
                "success": True,
                "error_message": "Action execution {} (execution id: {}) kicked off "
                "successfully.".format(execution_instance.action, execution.id),
            }

        return make_fake_result(result, True)

    def manage_resource(
        self, resource_type: str, name: str, pack: str, enabled: bool
    ) -> Dict[str, Optional[Union[bool, str]]]:
        # we use self.client.managers instead of self.client.<resource type> because
        # not all resources are available as properties on the client.

        resource = self.client.managers[resource_type].get_by_name(name=name, pack=pack)
        if resource is None:
            result = {
                "want_enabled": enabled,
                "enabled_before": None,
                "enabled_after": None,
                "error_message": f"{resource_type} {pack}.{name} not found.",
            }
            return result
        enabled_before = getattr(resource, "enabled", False)
        result = {
            "want_enabled": enabled,
            "enabled_before": enabled_before,
            "enabled_after": None,
            "error_message": "",
        }

        if self.check_mode:
            result["enabled_after"] = enabled
            return result

        try:
            resource.enabled = result["enabled_after"] = enabled
            self.client.managers[resource_type].update(resource)
        except Exception as exc:
            result["enabled_after"] = enabled_before
            result[
                "error_message"
            ] = f"Could not update {resource_type} {pack}.{name}: {exc}"
        return result


if __name__ == "__main__":
    # to run against test list of resources under localhost key in the pack_resources.yaml file
    # action = ManagePackResources(config={"local_test_with": "localhost"})
    action = ManagePackResources(config={})
    res = action.run(from_packs=["st2gitops"], check_mode=True)
    import pprint

    pp = pprint.PrettyPrinter(indent=2, width=200)
    pp.pprint(res)
