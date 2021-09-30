#date: 2021-09-30T17:03:04Z
#url: https://api.github.com/gists/41e45f2c3241e53601bf3d57cde388f6
#owner: https://api.github.com/users/hoshiyosan

"""
Defines how session with VROPS must be established.
"""

import logging
from datetime import datetime

import requests

from infrapi.clients.utils.apisession import APISession
from infrapi.clients.vrops.exceptions import VROPSAuthenticationError

LOGGER = logging.getLogger(__name__)


class VROPSSession(APISession):
    """
    Defines how session with VROPS must be established.
    Manage concatenation with base URL and authentication.
    """

    def __init__(self, username: str, password: str, base_url: str):
        super().__init__(base_url, verify_ssl=False)
        self.headers.update({"Accept": "application/json"})
        self.username = username
        self.password = password
        self.__token_validity = None

    def is_authenticated(self) -> bool:
        """
        Add token validity check to detect expired authentication before error
        """
        if self.__token_validity and self.__token_validity < datetime.utcnow():
            LOGGER.info("Token for VROPS session has expired")
            return False
        return super().is_authenticated()

    def authenticate(self):
        """
        Exchange credentials for an access token.
        """
        LOGGER.info("Authenticating on VROPS %s", self.base_url)
        response = requests.post(
            f'{self.base_url}/suite-api/api/auth/token/acquire', 
            headers={"Accept": "application/json"}, 
            json={
                "username": self.username,
                "password": self.password
            },
            verify=False,
        )
        if response.status_code != 200:
            raise VROPSAuthenticationError("Something bad happened.")
        
        response_data = response.json()
        # set authorization header for subsequent requests
        self.headers["Authorization"] = "vRealizeOpsToken " + response_data["token"]
        # store token validity to detect expired authentication before error
        self.__token_validity = datetime.utcfromtimestamp(response_data["validity"] / 1000)
        LOGGER.info("Authentication succeed with VROPS %s", self.base_url)
