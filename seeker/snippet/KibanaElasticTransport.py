#date: 2025-08-22T17:11:42Z
#url: https://api.github.com/gists/4d5247a4d72b2b08747eb3d36fe09adb
#owner: https://api.github.com/users/sifex

import base64
from urllib.parse import urlparse, quote

from elastic_transport import Transport, NodeConfig, AsyncTransport


class KibanaProxyTransport(Transport):
    """Simple transport that tunnels Elasticsearch requests via Kibana Dev Tools proxy."""

    def __init__(
        self,
        kibana_host: str,
        username: str,
        password: "**********"
        verify_certs: bool = True,
        **kwargs,
    ):
        """
        Initialize the Kibana proxy connection.

        Args:
            kibana_host: Kibana host URL (e.g., "https://my-kibana.example.com")
            username: Kibana username
            password: "**********"
            verify_certs: Whether to verify SSL certificates
        """
        # Parse the Kibana host to create a proper NodeConfig
        parsed_url = urlparse(kibana_host)

        # Create NodeConfig for Kibana host
        node_config = NodeConfig(
            scheme=parsed_url.scheme or "https",
            host=parsed_url.hostname,
            port=parsed_url.port or (443 if parsed_url.scheme == "https" else 80),
        )

        # Set up authentication headers
        credentials = f"{username}: "**********"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        self._auth_headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "x-elastic-internal-origin": "Kibana",
            "kbn-xsrf": "reporting",
            "accept": "application/json",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Initialize Transport with proper parameters
        super().__init__(
            node_configs=[node_config],
            **kwargs,
        )

    def perform_request(self, method: str, target: str, **kwargs):
        """
        Intercept Elasticsearch requests and proxy them through Kibana's dev tools endpoint.

        Converts: GET /_search
        To: POST /api/console/proxy?path=_search&method=GET
        """
        # Add auth headers to the request
        for key, value in self._auth_headers.items():
            kwargs["headers"][key] = value

        # Clean up the path and encode it
        es_path = target.lstrip("/")
        encoded_path = quote(es_path, safe="")

        # Convert to Kibana proxy format
        kibana_target = (
            f"/api/console/proxy?path={encoded_path}&method={method.upper()}"
        )

        # Forward as POST to Kibana proxy
        return super().perform_request("POST", kibana_target, **kwargs)


class AsyncKibanaProxyTransport(AsyncTransport):
    """Asynchronous transport that tunnels Elasticsearch requests via Kibana Dev Tools proxy."""

    def __init__(
        self, kibana_host: "**********": str, password: str, verify_certs: bool = True
    ):
        """
        Initialize the Kibana proxy connection.

        Args:
            kibana_host: Kibana host URL (e.g., "https://my-kibana.example.com")
            username: Kibana username
            password: "**********"
            verify_certs: Whether to verify SSL certificates
        """
        # Parse the Kibana host to create a proper NodeConfig
        parsed_url = urlparse(kibana_host)

        # Create NodeConfig for Kibana host
        node_config = NodeConfig(
            scheme=parsed_url.scheme or "https",
            host=parsed_url.hostname,
            port=parsed_url.port or (443 if parsed_url.scheme == "https" else 80),
        )

        # Set up authentication headers
        credentials = f"{username}: "**********"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        self._auth_headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "x-elastic-internal-origin": "Kibana",
            "kbn-xsrf": "reporting",
        }

        # Initialize AsyncTransport with proper parameters
        super().__init__(node_configs=[node_config])

    async def perform_request(self, method: str, target: str, **kwargs):
        """Intercept Elasticsearch requests and proxy them through Kibana's dev tools endpoint."""
        # Add auth headers to the request
        headers = kwargs.get("headers", {})
        headers.update(self._auth_headers)
        kwargs["headers"] = headers

        # Clean up the path and encode it
        es_path = target.lstrip("/")
        encoded_path = quote(es_path, safe="")

        # Convert to Kibana proxy format
        kibana_target = (
            f"/api/console/proxy?path={encoded_path}&method={method.upper()}"
        )

        # Forward as POST to Kibana proxy
        return await super().perform_request("POST", kibana_target, **kwargs)
