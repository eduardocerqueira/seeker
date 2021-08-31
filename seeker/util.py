from configparser import SafeConfigParser
import json


def get_config(section, parameter):
    config = SafeConfigParser()
    config.read("seeker.conf")
    return json.loads(config.get(section, parameter))
