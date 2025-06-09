#date: 2025-06-09T16:55:07Z
#url: https://api.github.com/gists/edcf0a2723ef91d0f52c09f4a97f1132
#owner: https://api.github.com/users/pearmaster

import inspect
import functools
from typing import Callable
from time import sleep

class MqttClient:
    """ Just a stand-in for an actual MQTT Client that implements just enough to
    show a principle. 
    """

    def __init__(self, hostname):
        self.hostname = hostname
        self.subscriptions = dict()

    def subscribe(self, topic: str, callback: Callable[str, str]):
        """ Adds a subscription.  When a message is received at the provided topic,
        the provided callback is called.
        """
        self.subscriptions[topic] = callback

    def inject_message(self, topic: str, payload: str):
        """ For testing and demonstration, calling this method stimulates the client
        in the same way as receiving a message.
        """
        print(f"Handling message to {topic}")
        if topic in self.subscriptions:
            callback = self.subscriptions[topic]
            callback(topic, payload)


class LedClient:

    def __init__(self, mqtt_client: MqttClient, receivers: dict[str, Callable[[str, str], None]], context: object|None):
        """ An LedClient is a domain specific client that communicates through MQTT.
        Just enough code is written to demonstrate a design principle.
        """
        self.mqtt_client = mqtt_client
        for topic, cb in receivers.items():
            if context is not None:
                callback = functools.partial(cb, context)
            else:
                callback = cb
            self.mqtt_client.subscribe(topic, callback)

    def set_state(self, new_state: bool):
        ...

class LedClientBuilder:
    """ This class captures callback methods by use of the `add_receiver` decorator.
    When all the callbacks are captured, an LedClient can be created and used.
    """
   
    def __init__(self):
        self.receivers: dict[str, Callable[[str, str], None]] = dict()

    def create(self, mqtt_client: MqttClient, context: object|None):
        """ With all the collected 
        """
        inst = LedClient(mqtt_client, self.receivers, context)
        return inst

    def receiver(self, topic: str):
        """ A "receiver" is a function that should be called when a message is received to
        the specified MQTT topic.  This is a decorator around the function.
        """
        def outer(func):
            @functools.wraps(func)
            def inner(*args, **kwargs):
                # Stuff could be done here if we wanted.
                return func(*args, **kwargs)
            self.receivers[topic] = inner
            return inner
        return outer
    
    def add_receiver(self, topic: str, callback: Callable[[str, str], None]):
        """ Manually add a receiver."""
        self.receivers[topic] = callback

class HardwareAbstraction:

    # This works within the context of the HardwareAbstraction class (before an instance)
    # is created.  When we create an instance, all the decorated methods are re-bound to the
    # instance.
    led_client_builder = LedClientBuilder()

    def __init__(self, mqtt_client):
        self.led_client = HardwareAbstraction.led_client_builder.create(mqtt_client, self)

    @led_client_builder.receiver('led/state')
    def receive_state(self, topic, payload):
        print(f"Received Led State: {payload}")

    def turn_on_led(self):
        self.led_client.set_state(True)

# And how, to try this outside of a class

another_led_client_builder = LedClientBuilder()

@another_led_client_builder.receiver("Hello/World")
def hello_receive(topic, payload):
        print(f"Received Hello: {payload}")

if __name__ == '__main__':
    mqtt_client = MqttClient('localhost')

    hal_one = HardwareAbstraction(mqtt_client)

    led_client = another_led_client_builder.create(mqtt_client, None)

    print("sleeping")
    sleep(1)

    mqtt_client.inject_message("led/state", '{"led_on":True}')
    mqtt_client.inject_message("Hello/World", "You're awesome")

