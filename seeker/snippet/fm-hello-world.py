#date: 2023-03-01T16:49:16Z
#url: https://api.github.com/gists/5d3f3d7c9b4aaf6798df5af7d2e887dd
#owner: https://api.github.com/users/markawm

# Import Rollout SDK
import os

from rox.server.rox_server import Rox
from rox.server.flags.rox_flag import RoxFlag
from rox.core.entities.rox_string import RoxString
from rox.core.entities.rox_int import RoxInt
from rox.core.entities.rox_double import RoxDouble

# Create Roxflags in the Flags container class
class Flags:
    def __init__(self):
        #Define the feature flags
        self.enableTutorial = RoxFlag(False)
        self.titleColors = RoxString('White', ['White', 'Blue', 'Green', 'Yellow'])
        self.page = RoxInt(1, [1, 2, 3])
        self.percentage = RoxDouble(99.9, [10.5, 50.0, 99.9])


def user_unhandled_error_invoker(errorDetails):
    print('Unhandled error. trigger type: {}, source: {}, exception: {}.'.format(errorDetails.exception_trigger, errorDetails.exception_source, errorDetails.exception))


envKey = os.getenv('FM_ENVIRONMENT_KEY')
print('Using envKey', envKey)

flags = Flags()

# Register the flags container with Rollout
Rox.register(flags)

Rox.set_userspace_unhandled_error_handler(user_unhandled_error_invoker)

# Setup the Rollout environment key
cancel_event = Rox.setup(envKey).result();

# Boolean flag example
print('enableTutorial is %s' % flags.enableTutorial.is_enabled())

# string flag example
print('color is {}'.format(flags.titleColors.get_value()))

# int flag example
print('page is {}'.format(flags.page.get_value()))

# double flag example
print('percentage is {}'.format(flags.percentage.get_value()))