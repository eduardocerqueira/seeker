#date: 2022-12-23T17:03:06Z
#url: https://api.github.com/gists/bd598f956ae55e6df4d3ac980557634e
#owner: https://api.github.com/users/DanEdens

import os
import subprocess

def subscribe_to_mqtt_topic(
    topic: str, path: str, awsip: str=os.getenv("AWSIP"), 
    awsport: str=os.getenv("AWSPORT")) -> str:
    """
    Subscribes to the MQTT topic with the specified name and writes the received messages to a file with the same name.
    Returns the file path of the created file.
    """
    # Replace any equals signs in the name with colons
    topic = topic.replace("=", ":")

    # Get the last character of the name
    last = topic[-1]

    # Subscribe to the MQTT topic and write the received messages to a file with the same name
    subprocess.run([
        "mosquitto_sub", "-h", awsip, "-p", awsport, 
        "-t", f"commands/{topic}", "-C", "1"], 
        stdout=open(f"{path}/{last}.cmd", "w")
        )

    # Return the file path
    return f"{path}/{last}.cmd"
