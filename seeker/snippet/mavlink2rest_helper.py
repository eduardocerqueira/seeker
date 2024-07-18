#date: 2024-07-18T16:56:22Z
#url: https://api.github.com/gists/2583c221cfd22708b77ec1ae5d481c8d
#owner: https://api.github.com/users/Williangalvani

import json
import time
from typing import Any, Optional

import requests
from loguru import logger

GPS_GLOBAL_ORIGIN_ID = 49

# pylint: disable=too-many-instance-attributes
class Mavlink2RestHelper:
    """
    Responsible for interfacing with Mavlink2Rest
    """

    def __init__(self, m2r_url: str, vehicle: int = 1, component: int = 1):
        # store vehicle and component to access telemetry data from
        self.url = m2r_url
        self.vehicle = vehicle
        self.component = component
        self.rc_channels_override_template = """
    {{
  "header": {{
    "system_id": 255,
    "component_id": 0,
    "sequence": 0
  }},
  "message": {{
    "type": "RC_CHANNELS_OVERRIDE",
    "chan1_raw": {0},
    "chan2_raw": {1},
    "chan3_raw": {2},
    "chan4_raw": {3},
    "chan5_raw": {4},
    "chan6_raw": {5},
    "chan7_raw": {6},
    "chan8_raw": {7},
    "chan9_raw": {8},
    "chan10_raw":{9},
    "chan11_raw":{10},
    "chan12_raw":{11},
    "chan13_raw":{12},
    "chan14_raw":{13},
    "chan15_raw":{14},
    "chan16_raw":{15},
    "chan17_raw":0,
    "chan18_raw":0,
    "target_system": 1,
    "target_component": 1
  }}
}}
"""

    def set_param(self, param_name, param_type, param_value):
        """
        Sets parameter "param_name" of type param_type to value "value" in the autpilot
        Returns True if succesful, False otherwise
        """
        try:
            data = json.loads(requests.get(self.url + "/v1/helper/mavlink?name=PARAM_SET").text)

            for i, char in enumerate(param_name):
                data["message"]["param_id"][i] = char

            data["message"]["param_type"] = {"type": param_type}
            data["message"]["param_value"] = param_value

            result = requests.post(self.url + "/mavlink", json=data)
            return result.status_code == 200
        except Exception as error:
            logger.warning(f"Error setting parameter '{param_name}': {error}")
            return False

    def send_rc_override(self, channel):
        """
        Sends STATUSTEXT message to the GCS
        """
        # make sure we have at least 16 channels
        # center the remaining ones
        if len(channel) < 16:
            channel = channel + [1500] * (16 - len(channel))
        
        data = json.loads(self.rc_channels_override_template.format(*channel))
        try:
            requests.post(self.url + "/mavlink", json=data)
        except Exception as error:
            logger.warning(f"Error sending RC_CHANNELS_OVERRIDE: {error}")
            return False

