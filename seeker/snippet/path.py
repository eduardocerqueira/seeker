#date: 2023-11-15T16:51:27Z
#url: https://api.github.com/gists/25f59a0644232dd9bad76dfc4d39d011
#owner: https://api.github.com/users/geox25

# Here we import our own MQTT library which takes care of a lot of boilerplate
# code related to connecting to the MQTT server and sending/receiving messages.
# It also helps us make sure that our code is sending the proper payload on a topic
# and is receiving the proper payload as well.
from bell.avr.mqtt.client import MQTTModule
from bell.avr.mqtt.payloads import AvrAutonomousEnablePayload

# This imports the third-party Loguru library which helps make logging way easier
# and more useful.
# https://loguru.readthedocs.io/en/stable/
from loguru import logger

import time

# This creates a new class that will contain multiple functions
# which are known as "methods". This inherits from the MQTTModule class
# that we imported from our custom MQTT library.
class Sandbox(MQTTModule):
    # The "__init__" method of any class is special in Python. It's what runs when
    # you create a class like `sandbox = Sandbox()`. In here, we usually put
    # first-time initialization and setup code. The "self" argument is a magic
    # argument that must be the first argument in any class method. This allows the code
    # inside the method to access class information.
    def __init__(self) -> None:
        # This calls the original `__init__()` method of the MQTTModule class.
        # This runs some setup code that we still want to occur, even though
        # we're replacing the `__init__()` method.
        super().__init__()
        # Here, we're creating a dictionary of MQTT topic names to method handles.
        # A dictionary is a data structure that allows use to
        # obtain values based on keys. Think of a dictionary of state names as keys
        # and their capitals as values. By using the state name as a key, you can easily
        # find the associated capital. However, this does not work in reverse. So here,
        # we're creating a dictionary of MQTT topics, and the methods we want to run
        # whenever a message arrives on that topic.
        self.topic_map = {"avr/autonomous/enable": self.auton}

    def auton(self, payload: AvrAutonomousEnablePayload) -> None:
        enabled = payload["enabled"]
        msg = "null"

        if enabled == True:
            msg = "ENABLED"
        elif enabled == False:
            msg = "DISABLED"

        logger.success("AUTONOMY " + msg)

        if enabled == True:
            self.path()
            # logger.info("DISARM")
            # self.send_message("avr/fcm/disarm", {})
        elif enabled == False:
            logger.info("LANDING")
            box.send_message("avr/fcm/land", {})

            logger.info("DISARM")
            self.send_message("avr/fcm/disarm", {}) 

    def path(self) -> None:
        logger.info("CAPTURING HOME & ARM")
        self.send_message("avr/fcm/capture_home", {})
        self.send_message("avr/fcm/arm", {})

        logger.info("UPLOADING MISSION")
        """
        self.send_message("avr/fcm/upload_mission", {
            "waypoints": [
                { # takeoff
                    "type": "takeoff"
                },
                {
                    "type": "land"
                }
            ]
        })
        """
        logger.info("STARTING MISSION")
        # self.send_message("avr/fcm/start_mission", {})
        box.send_message("avr/fcm/takeoff", {"alt": 3.12})

if __name__ == "__main__":
    # This is what actually initializes the Sandbox class, and executes it.
    # This is nested under the above condition, as otherwise, if this file
    # were imported by another file, these lines would execute, as the interpreter
    # reads and executes the file top-down. However, whenever a file is called directly
    # with `python file.py`, the magic `__name__` variable is set to "__main__".
    # Thus, this code will only execute if the file is called directly.
    box = Sandbox()
    # The `run` method is defined by the inherited `MQTTModule` class and is a
    # convience function to start processing incoming MQTT messages infinitely.
    box.run_non_blocking()

    logger.debug("Printing Works")

    while True:
        time.sleep(0.1)
