#date: 2023-12-20T16:48:03Z
#url: https://api.github.com/gists/bc86396a1424b29ef25ed5c3eaa5aca1
#owner: https://api.github.com/users/nice-switch

import odv, sys

# Checks for available args and grabs the first one if available.
# NOTE sys.argv[0] is the file arg.
target_service: str | None = len(sys.argv) > 1 and sys.argv[1] or None

# Validity check for target_service, this should always be something!
if (target_service is None):
    # TODO add available args list with error message.
    raise Exception("No target service to initialize...")

# Just some post processing on the string. UPPER -> lower.
target_service = target_service.lower()

# Looks for more args after target_service, when more are available it makes a new list with the args, or makes an empty list when none are available.
cli_args: list[str] = len(sys.argv) > 2 and sys.argv[3::] or []

# TODO Make a TOML/JSON file for all configuration.
match target_service:
    case odv.enum.ServiceType.DATABASE:

        odv.initialize_database_service(
            database_type=odv.enum.DatabaseType.DEVELOPMENT
        )
        
        # TODO add args for publish that use the config files.
        odv.publish_database_service(
            api_port=odv.enum.DatabasePort.DEVELOPMENT,
            api_host=odv.enum.DatabaseHost.LOCALHOST
        )
        
    case odv.enum.ServiceType.CLIENT:
        print("Starting client service...")
    case odv.enum.ServiceType.GUARD:
        print("Starting guard service...")
    case odv.enum.ServiceType.WEB:
        print("Starting web service...")