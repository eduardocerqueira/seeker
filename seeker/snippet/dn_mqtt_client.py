#date: 2022-07-07T17:00:39Z
#url: https://api.github.com/gists/8ad6669070e8ceb59f51c8d07aa931b7
#owner: https://api.github.com/users/shahrukhx01

"""
Documentation:
Install the paho mqtt package to run this script using the following command:

pip install paho-mqtt

Run the script using the following command:

python dn_mqtt_client.py
"""
import paho.mqtt.client as mqtt


COMMAND_REPLIES = { "CMD1" : "Apple", "CMD2" : "Cat", "CMD3" : "Dog" ,
"CMD4" : "Rat" , "CMD5" : "Boy" , "CMD6" : "Girl" ,
"CMD7" : "Toy" }

STUDENT_ID = "7004431"

def reply_command(client, topic, command):
    """
    To send replies for specific commands
    """

    client.subscribe(topic)
    print(f"Command received: {command}")
    print(f"Command reply: {COMMAND_REPLIES[command]}")
    client.publish(topic, COMMAND_REPLIES[command])

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Broker connected!")
        client.subscribe("login")
        client.subscribe(f"{STUDENT_ID}/UUID")
        client.publish("login", STUDENT_ID)

def on_message(client, userdata, message):
    if message.topic == f"{STUDENT_ID}/UUID":
        print("Login completed.")
        message_content = str(message.payload.decode("utf-8"))
        print(f"UUID: {message_content}")
        client.subscribe(f"{message_content}")

    elif "CMD" in str(message.payload.decode("utf-8")):
        command = str(message.payload.decode("utf-8"))
        reply_command(client=client, topic=f"{message.topic}/{command}",
        command=command)

    elif "Well done my IoT!" == str(message.payload.decode("utf-8")):
        print("Final message:" , str(message.payload.decode("utf-8")))
        client.loop_stop()    #Stop loop
        client.disconnect() # disconnect
        print("Client disconnected, BYE BYE!")


# client ID "dn-assignment-12"
client = mqtt.Client("dn-assignment-12")
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set("shkh00001@stud.uni-saarland.de", "7004431")
client.connect('inet-mqtt-broker.mpi-inf.mpg.de', 1883)

client.loop_forever()  # Start networking daemon
