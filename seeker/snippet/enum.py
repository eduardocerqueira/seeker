#date: 2023-12-20T16:48:31Z
#url: https://api.github.com/gists/f5d63818d15138052dbad2fc1c25b767
#owner: https://api.github.com/users/nice-switch

import socket

from enum import StrEnum, IntEnum
    
    
class ServiceType(StrEnum):
    """Types of services that are available to use."""
    DATABASE = "database"
    CLIENT = "client"
    GUARD = "guard"
    WEB = "web"


class DatabaseType(StrEnum):
    """Types of databases that are available to use."""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    #EXPERIMENTAL = TODO load workspace/database.json


# TODO add config loader for DatabaseHost & DatabasePort    
class DatabaseHost(StrEnum):
    """Available IPs to host the API."""
    MACHINE = "127.0.0.1"
    # NOTE this is potentionally unsafe, will be looking more into this.
    #socket.gethostbyname_ex(socket.gethostname())[2][-1::][0]
    LOCALHOST = socket.gethostbyname(socket.gethostname())
    #CONFIG = TODO load workspace/database.json


class DatabasePort(IntEnum):
    """Available ports for the API."""
    PRODUCTION = 6000
    DEVELOPMENT = 6500
    #PRODUCTION_CONFIG = TODO load workspace/database.json
    #DEVELOPMENT_CONFIG = TODO load workspace/database.json 


