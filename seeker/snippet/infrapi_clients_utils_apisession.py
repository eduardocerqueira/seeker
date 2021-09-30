#date: 2021-09-30T17:03:04Z
#url: https://api.github.com/gists/41e45f2c3241e53601bf3d57cde388f6
#owner: https://api.github.com/users/hoshiyosan

"""
Defines base class APISession that is an extended requests.Session
that provide a more convenient session to call an API.
"""
import logging
import urllib3

import requests


LOGGER = logging.getLogger(__name__)


class APISession(requests.Session):
    """
    Generic class to provide a more convenient session to call an API.
    """

    def __init__(self, base_url: str, verify_ssl=True):
        super().__init__()
        self.base_url = base_url.rstrip('/')  # remove trailing / if exists
        self.__authenticated = False
        self.__verify_ssl = verify_ssl

        if not verify_ssl:
            # don't log insecure requests if all requests are supposed to be insecure
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def url_for(self, url_or_path: str) -> str:
        """
        Compute absolute URL given a path or an URL.
        """
        # url is absolute
        if url_or_path.startswith("http"):
            return url_or_path
        # url is relative, make it absolute
        return self.base_url + '/' + url_or_path.lstrip('/')

    def is_authenticated(self) -> bool:
        """
        Method to check if authentication has been performed / is still valid.
        """
        return self.__authenticated

    def request(self, method: str, uri: str, *args, **kwargs):  # pylint: disable=arguments-differ
        """
        Override requests.Session method used internally by get, put, post, delete methods.
        in order to avoid specifying absolute URLs and perform authentication.
        """
        if not self.is_authenticated():
            self.authenticate()
            self.__authenticated = True

        verify_ssl = kwargs.pop("verify", None)
        if verify_ssl is None:
            verify_ssl = self.__verify_ssl

        LOGGER.debug("[request] %s %s", method, self.url_for(uri))
        response = super().request(method, self.url_for(uri), *args, verify=verify_ssl, **kwargs)
        LOGGER.debug("[response] %s", response.status_code)
        return response

    def authenticate(self):
        """
        Override this method in your subclass to define how authentication is performed.
        It defaults to have no effect...
        """
