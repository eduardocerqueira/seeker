#date: 2026-03-05T18:29:53Z
#url: https://api.github.com/gists/69a94ff7883bb05a9043e3b0e4d5bd25
#owner: https://api.github.com/users/christophertubbs

"""
@author: Christopher Tubbs

Utility for generating a reasonably complete Django ALLOWED_HOSTS list
without embedding deployment-specific values into source code.
"""
__all__ = ["build_allowed_hosts"]

import os
import socket
import typing
import collections.abc as generic


APP_PREFIX: typing.Final[str] = ""
"""Add a prefix to isolate app-specific environment variables"""
KEY_SEPARATOR: str = "_"
"""
A separator character to use to separate domain parts in keys for settings. Example: '_' for os environment variables, 
':' for redis environment variables
"""
VALUE_SEPARATOR: str = ","
"""
A separator character to use to separate values in a list of values from a setting
"""

ALLOWED_HOSTS_KEY: typing.Final[str] = f'{APP_PREFIX + KEY_SEPARATOR if APP_PREFIX else ""}ALLOWED_HOSTS'
"""The key for a variable holding a list of explicitly allowed host names"""
ALLOW_SUBDOMAINS_KEY: typing.Final[str] = f'{ALLOWED_HOSTS_KEY}{KEY_SEPARATOR}ALLOW_SUBDOMAINS'
"""The key for the setting that dictates whether a subdomain for localhost may be used"""
INCLUDE_FQDN_KEY: typing.Final[str] = f'{ALLOWED_HOSTS_KEY}{KEY_SEPARATOR}INCLUDE_FQDN'
"""The key for the setting that dictates whether to include the fully qualified domain name"""
INCLUDE_LOCAL_ADDRESSES_KEY: typing.Final[str] = f'{ALLOWED_HOSTS_KEY}{KEY_SEPARATOR}INCLUDE_LOCAL_ADDRESSES'
"""
The key for the setting that dictates whether to include the local IP address from the self referenced socket
"""
INCLUDE_HOST_NAME_KEY: typing.Final[str] = f'{ALLOWED_HOSTS_KEY}{KEY_SEPARATOR}INCLUDE_HOST_NAME'
"""The key for the setting that dictates whether to include the socket hostname"""
INCLUDE_ENVIRONMENT_VARIABLES_KEY: typing.Final[str] = (
    f'{ALLOWED_HOSTS_KEY}{KEY_SEPARATOR}INCLUDE_ENVIRONMENT_VARIABLES'
)
"""
The settings key that may dictate whether to include hosts from a handful of environment variables
"""

AccessFunction = generic.Callable[[str, typing.Any], str | bytes | None]
"""
A function used to access a setting.
Provides a common interface for 'get' functions from either os environment variables or cache access like 'get' 
from a redis instance
"""

DEFAULT_ACCESS_FUNCTION: AccessFunction = os.environ.get
"""
The default function to use when selecting a value. Replace with something like `redis.Redis(**kwargs).get` in order 
to access another source of values
"""


def get_supported_allowed_hosts_environment_variables(
    *,
    access_function: AccessFunction = None
) -> dict[str, str]:
    """
    Retrieves and processes supported environment variables related to allowed hosts and subdomains.

    This function accesses predefined environment variables using the provided access
    function. If no access function is provided, a default one is used. The returned
    dictionary contains the corresponding keys and their processed string values.
    Environment variable values that are in bytes are decoded to strings. Variables
    with no values are assigned an empty string.

    :param access_function: A callable used to retrieve environment variable values.
                            Defaults to `DEFAULT_ACCESS_FUNCTION` if not provided.
    :type access_function: AccessFunction, optional
    :return: A dictionary with keys representing environment variable names and values
             as their corresponding processed string values.
    :rtype: dict[str, str]
    """
    if access_function is None:
        access_function = DEFAULT_ACCESS_FUNCTION

    keys_and_values: dict[str, str] = {}

    allow_hosts_value: str | bytes | None = access_function(ALLOWED_HOSTS_KEY)

    if not allow_hosts_value:
        allow_hosts_value = ""

    if isinstance(allow_hosts_value, bytes):
        allow_hosts_value = allow_hosts_value.decode()

    keys_and_values[ALLOWED_HOSTS_KEY] = typing.cast(str, allow_hosts_value)

    allow_subdomains_value: str | bytes | None = access_function(ALLOW_SUBDOMAINS_KEY)

    if not allow_subdomains_value:
        allow_subdomains_value = ""

    if isinstance(allow_subdomains_value, bytes):
        allow_subdomains_value = allow_subdomains_value.decode()

    keys_and_values[ALLOW_SUBDOMAINS_KEY] = typing.cast(str, allow_subdomains_value)

    include_fqdn_value: str | bytes | None = access_function(INCLUDE_FQDN_KEY)

    if not include_fqdn_value:
        include_fqdn_value = ""

    if isinstance(include_fqdn_value, bytes):
        include_fqdn_value = include_fqdn_value.decode()

    keys_and_values[INCLUDE_FQDN_KEY] = typing.cast(str, include_fqdn_value)

    include_local_addresses_value: str | bytes | None = access_function(INCLUDE_LOCAL_ADDRESSES_KEY)
    if isinstance(include_local_addresses_value, bytes):
        include_local_addresses_value = include_local_addresses_value.decode()

    if not include_local_addresses_value:
        include_local_addresses_value = ""

    keys_and_values[INCLUDE_LOCAL_ADDRESSES_KEY] = typing.cast(str, include_local_addresses_value)

    include_host_name_value: str | bytes | None = access_function(INCLUDE_HOST_NAME_KEY)
    if not include_host_name_value:
        include_host_name_value = ""

    if isinstance(include_host_name_value, bytes):
        include_host_name_value = include_host_name_value.decode()

    keys_and_values[INCLUDE_HOST_NAME_KEY] = typing.cast(str, include_host_name_value)

    include_environment_variables_value: str | bytes | None = access_function(INCLUDE_ENVIRONMENT_VARIABLES_KEY)

    if not include_environment_variables_value:
        include_environment_variables_value = ""

    if isinstance(include_environment_variables_value, bytes):
        include_environment_variables_value = include_environment_variables_value.decode()

    keys_and_values[INCLUDE_ENVIRONMENT_VARIABLES_KEY] = typing.cast(str, include_environment_variables_value)

    return keys_and_values


def _environment_variable_is_true(
    key: str,
    default: str = "false",
    *,
    access_function: AccessFunction = None
) -> bool:
    """
    Determines whether an environment variable is set to a truthy value. This function checks
    the value of the given environment variable and evaluates if it corresponds to a truthy
    value such as "true", "1", "on", "t", "y", or "yes" (case-insensitive). If the variable
    is not found, a default value is used.

    :param key: The name of the environment variable to check.
    :type key: str
    :param default: The default value to use if the environment variable is not set.
                    Defaults to "false".
    :type default: str
    :param access_function: Function used to access the environment variable. If not provided,
                            a default access function is used.
    :type access_function: AccessFunction, optional
    :return: True if the environment variable is set to a truthy value, False otherwise.
    :rtype: bool
    """
    if access_function is None:
        access_function = DEFAULT_ACCESS_FUNCTION

    value: str | bytes | None = access_function(
        key=key,
        default=default
    )

    if value is None:
        value = default

    if not value:
        return False

    if isinstance(value, bytes):
        value: str = value.decode()

    return value.lower() in ('true', '1', 'on', 't', 'y', 'yes')


def _get_local_ip_addresses() -> generic.Iterable[str]:
    """
    Get all local IP addresses of the current machine.

    This function resolves and collects all local IP addresses, including those
    associated with the hostname as well as the primary outbound IP determined
    via UDP socket connection. If any errors occur during these operations, such
    errors are silently ignored, and no exceptions are propagated to the caller.

    :return: A set of IP addresses as strings associated with the current machine.
    :rtype: generic.Iterable[str]
    """
    ip_addresses: set[str] = set()

    try:
        hostname: str = socket.gethostname()
        resolved: list[str] = socket.gethostbyname_ex(hostname)[2]
        ip_addresses.update(resolved)
    except socket.error:
        pass

    # UDP trick to discover primary outbound IP
    try:
        udp_socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.connect(("8.8.8.8", 80))
        ip_addresses.add(udp_socket.getsockname()[0])
        udp_socket.close()
    except OSError:
        pass

    return ip_addresses


def get_hosts_from_environment(
    environment_hostnames_key: str = ALLOWED_HOSTS_KEY,
    *,
    access_function: AccessFunction = None,
    value_separator: str = None
) -> generic.Iterable[str]:
    """
    Extract a list of hostnames from environment variables based on provided parameters. The function retrieves
    a raw value using a specified access function and parses it using a defined separator.

    :param environment_hostnames_key: The environmental variable key to look up for hostnames.
    :param access_function: A callable used to access the value from the environment. Defaults to a predefined
        access function if not provided.
    :param value_separator: The separator used to split the hostnames string. Defaults to a predefined value
        if not specified.
    :return: A list of parsed hostnames derived from the environmental variable, split using the given
        separator.
    """
    if value_separator is None:
        value_separator = VALUE_SEPARATOR

    if access_function is None:
        access_function = DEFAULT_ACCESS_FUNCTION

    hostnames_from_environment: str | bytes | None = access_function(environment_hostnames_key, "")

    if not hostnames_from_environment:
        hostnames_from_environment: str = ""

    if isinstance(hostnames_from_environment, bytes):
        hostnames_from_environment: str = hostnames_from_environment.decode()

    return hostnames_from_environment.split(value_separator)


def build_allowed_hosts(
    *hosts: str,
    allow_subdomains: bool = None,
    include_fqdn: bool = None,
    include_environment_variables: bool = None,
    include_local_addresses: bool = None,
    include_host_name: bool = None,
    extra_hosts: typing.Iterable[str] | None = None,
    access_function: AccessFunction = None,
    value_separator: str = None,
) -> generic.Sequence[str]:
    """
    Builds and returns a list of allowed hostnames or IP addresses for a specific use case. The function
    provides a flexible way to construct a list of hosts by considering subdomains, hostnames, Fully
    Qualified Domain Names (FQDN), network addresses, and environment variables. It also accounts for
    explicitly defined hosts through its parameters.

    The feature set of this function allows customized constraints and inclusions based on the options
    provided, such as subdomain support, local network or environment-based addresses, and the inclusion
    of additional statically defined hosts.

    :param hosts: Arbitrary length list of hostnames or IPs to include in the allowed hosts. If
        subdomain support is enabled, subdomain variants of these values are also included.
    :param allow_subdomains: If True, subdomains of all provided hosts are allowed. Defaults to None,
        which infers the value from the relevant environmental variable.
    :param include_fqdn: If True, the local machine's fully qualified domain name (FQDN) is added to the
        list of allowed hosts. Defaults to None, which infers the value from the relevant environmental
        variable.
    :param include_environment_variables: If True, hostnames defined in environment variables (e.g.,
        container DNS names) are included. Defaults to None, which infers the value from the relevant
        environmental variable.
    :param include_local_addresses: If True, local network IP addresses detected on the machine are
        included in the list of allowed hosts. Defaults to None, which infers the value from the
        relevant environmental variable.
    :param include_host_name: If True, the local machine’s hostname is added to the list of allowed
        hosts. Defaults to None, which infers the value from the relevant environmental variable.
    :param extra_hosts: An iterable of additional hostnames or IPs explicitly added to the allowed hosts.
    :param access_function: A callable used to access environment variables, allowing customization of
        how environment variables are retrieved. Defaults to a pre-defined default access function.
    :param value_separator: A string that serves as the delimiter when parsing environment variables
        containing multiple hosts. Defaults to a pre-configured value.

    :return: A sorted list of unique allowed hostnames or IP addresses as strings. Some host entries
        will be derived based on the input parameters and system/environment conditions.
    """
    if access_function is None:
        access_function = DEFAULT_ACCESS_FUNCTION

    if value_separator is None:
        value_separator = VALUE_SEPARATOR

    if allow_subdomains is None:
        allow_subdomains = _environment_variable_is_true(
            ALLOW_SUBDOMAINS_KEY,
            default="true",
            access_function=access_function
        )

    if include_environment_variables is None:
        include_environment_variables = _environment_variable_is_true(
            INCLUDE_ENVIRONMENT_VARIABLES_KEY,
            default="true",
            access_function=access_function
        )

    if include_host_name is None:
        include_host_name = _environment_variable_is_true(
            INCLUDE_HOST_NAME_KEY,
            default="true",
            access_function=access_function
        )

    if include_fqdn is None:
        include_fqdn = _environment_variable_is_true(
            INCLUDE_FQDN_KEY,
            default="true",
            access_function=access_function
        )

    if include_local_addresses is None:
        include_local_addresses = _environment_variable_is_true(
            INCLUDE_LOCAL_ADDRESSES_KEY,
            default="true",
            access_function=access_function
        )

    allowed_hosts: set[str] = {
        f"{'.' if allow_subdomains else ''}localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        *hosts
    }

    if include_host_name:
        try:
            allowed_hosts.add(socket.gethostname())
        except socket.error:
            pass

    if include_fqdn:
        try:
            allowed_hosts.add(socket.getfqdn())
        except socket.error:
            pass

    if include_local_addresses:
        # Local network addresses
        local_addresses: generic.Iterable[str] = _get_local_ip_addresses()
        allowed_hosts.update(local_addresses)

    if include_environment_variables:
        # Docker / container hostnames
        environment_hostname: str | None = os.environ.get("HOSTNAME")
        if environment_hostname:
            allowed_hosts.add(environment_hostname)

        hosts_from_environment: generic.Iterable[str] = get_hosts_from_environment(
            access_function=access_function,
            value_separator=value_separator,
        )

        allowed_hosts.update(hosts_from_environment)

    # Explicit extras
    if extra_hosts:
        allowed_hosts.update(extra_hosts)

    return sorted(host for host in allowed_hosts if host)


# Add in functionality to call directory as an example/test
if __name__ == "__main__":
    def main(*args) -> int:
        supported_environment_variables: generic.Mapping[str, str] = get_supported_allowed_hosts_environment_variables(
            access_function=DEFAULT_ACCESS_FUNCTION
        )
        formatted_environment_variables: typing.Iterable[str] = map(
            lambda pair: f"{pair[0]}={pair[1]}",
            supported_environment_variables.items()
        )
        print(
            f"Supported environment variables:{os.linesep}"
            f"    - {(os.linesep + '    - ').join(formatted_environment_variables)}"
        )
        allowed_hosts: generic.Sequence[str] = build_allowed_hosts(*args)
        print(
            f"Allowed Hosts:{os.linesep}"
            f"    - {(os.linesep + '    - ').join(allowed_hosts)}"
        )
        return 0

    import sys
    exit_code: int = 0
    try:
        exit_code = main(*sys.argv[1:])
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as exc:
        print(f"An error occurred: {exc}")
        exit_code = 1
    sys.exit(exit_code)
