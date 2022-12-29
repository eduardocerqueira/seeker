#date: 2022-12-29T16:39:45Z
#url: https://api.github.com/gists/3b9fed447a7aa51effbc8688177e1097
#owner: https://api.github.com/users/waltlillyman

# Set variables from values defined in the OS's environment.
from os import getenv  # Environment variables

# Set defaults in case an environment variable is not defined :
default_team_id = 19
default_ha_host = 'homeassistant'
default_ha_port = 8123
default_webhook_id = 'press_nhl_goal_button'
default_log_level = 'DEBUG'

# When an env var is not set at all, its value is "None". When it's set with no value, it's value is ''. 
# Make sure to cast numeric environment variable values as int for later comparison.
team_id = int(getenv('TEAM_ID')) if (getenv('TEAM_ID') != None and getenv('TEAM_ID') != '') else default_team_id
ha_host = getenv('HA_HOST') if (getenv('HA_HOST') != None and getenv('HA_HOST') != '') else default_ha_host
ha_port = int(getenv('HA_PORT')) if (getenv('HA_PORT') != None and getenv('HA_PORT') != '') else default_ha_port
webhook_id = getenv('WEBHOOK_ID') if (getenv('WEBHOOK_ID') != None and getenv('WEBHOOK_ID') != '') else default_webhook_id
log_level = getenv('LOG_LEVEL') if (getenv('LOG_LEVEL') != None and getenv('LOG_LEVEL') != '') else default_log_level
