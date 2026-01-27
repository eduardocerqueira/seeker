#date: 2026-01-27T17:23:05Z
#url: https://api.github.com/gists/b91f47cb0620b537998903aab0f978b4
#owner: https://api.github.com/users/Ajak58a

from awsiot import mqtt5_client_builder
from awscrt import mqtt5
import threading, time

# Required Arguments
aws_endpoint = '<aws-endpoint>'
cert_path = r"C:\Users\ASUS\certs\Device-Certificate.crt"
private_key = r"C:\Users\ASUS\certs\Private.key"
client_id = 'Node3-PC'
TIMEOUT = 100

# Events used within callbacks to progress sample
connection_success_event = threading.Event()
stopped_event = threading.Event()
received_all_event = threading.Event()
received_count = 0


# Callback when any publish is received
def on_publish_received(publish_packet_data):
    publish_packet = publish_packet_data.publish_packet
    print("==== Received message from topic '{}': {} ====\n".format(
        publish_packet.topic, publish_packet.payload.decode('utf-8')))

    # Track number of publishes received
    global received_count
    received_count += 1
    if received_count == count:
        received_all_event.set()

# Callback for the lifecycle event Stopped
def on_lifecycle_stopped(lifecycle_stopped_data: mqtt5.LifecycleStoppedData):
    print("Lifecycle Stopped\n")
    stopped_event.set()

# Callback for lifecycle event Attempting Connect
def on_lifecycle_attempting_connect(lifecycle_attempting_connect_data: mqtt5.LifecycleAttemptingConnectData):
    print("Lifecycle Connection Attempt\nConnecting to endpoint: '{}' with client ID'{}'".format(
        aws_endpoint, client_id))

# Callback for the lifecycle event Connection Success
def on_lifecycle_connection_success(lifecycle_connect_success_data: mqtt5.LifecycleConnectSuccessData):
    connack_packet = lifecycle_connect_success_data.connack_packet
    print("Lifecycle Connection Success with reason code:{}\n".format(
        repr(connack_packet.reason_code)))
    connection_success_event.set()

# Callback for the lifecycle event Connection Failure
def on_lifecycle_connection_failure(lifecycle_connection_failure: mqtt5.LifecycleConnectFailureData):
    print("Lifecycle Connection Failure with exception:{}".format(
        lifecycle_connection_failure.exception))

# Callback for the lifecycle event Disconnection
def on_lifecycle_disconnection(lifecycle_disconnect_data: mqtt5.LifecycleDisconnectData):
    print("Lifecycle Disconnected with reason code:{}".format(
        lifecycle_disconnect_data.disconnect_packet.reason_code if lifecycle_disconnect_data.disconnect_packet else "None"))

if __name__ == '__main__':
    print("\nStarting MQTT5 X509 PubSub Sample\n")
    pub_message_count = 1
    pub_message_topic = 'node3/pub'
    pub_message_string = 'Welcome! This is a message from Operator. Thank You.'

    # Create MQTT5 client using mutual TLS via X509 Certificate and Private Key
    print("==== Creating MQTT5 Client ====\n")
    client = mqtt5_client_builder.mtls_from_path(
        endpoint=aws_endpoint,
        cert_filepath=cert_path,
        pri_key_filepath=private_key,
        on_publish_received=on_publish_received,
        on_lifecycle_stopped=on_lifecycle_stopped,
        on_lifecycle_attempting_connect=on_lifecycle_attempting_connect,
        on_lifecycle_connection_success=on_lifecycle_connection_success,
        on_lifecycle_connection_failure=on_lifecycle_connection_failure,
        on_lifecycle_disconnection=on_lifecycle_disconnection,
        client_id=client_id)
    
    # Start the client, instructing the client to desire a connected state. The client will try to 
    # establish a connection with the provided settings. If the client is disconnected while in this 
    # state it will attempt to reconnect automatically.
    print("==== Starting client ====")
    client.start()

    # We await the `on_lifecycle_connection_success` callback to be invoked.
    if not connection_success_event.wait(TIMEOUT):
        raise TimeoutError("Connection timeout")

    # Publish
    if pub_message_count == 0:
        print("==== Sending messages until program killed ====\n")
    else:
        print("==== Sending {} message(s) ====\n".format(pub_message_count))

    publish_count = 1
    while (publish_count <= pub_message_count) or (pub_message_count == 0):
        message = f"{pub_message_string} [{publish_count}]"
        print(f"Publishing message to topic '{pub_message_topic}': {message}")
        publish_future = client.publish(mqtt5.PublishPacket(
            topic=pub_message_topic,
            payload=message,
            qos=mqtt5.QoS.AT_LEAST_ONCE
        ))
        publish_completion_data = publish_future.result(TIMEOUT)
        print("PubAck received with {}\n".format(repr(publish_completion_data.puback.reason_code)))
        time.sleep(1.5)
        publish_count += 1

    received_all_event.wait(TIMEOUT)
    print("{} message(s) received.\n".format(received_count))

    # Stop the client. Instructs the client to disconnect and remain in a disconnected state.
    print("==== Stopping Client ====")
    client.stop()

    if not stopped_event.wait(TIMEOUT):
        raise TimeoutError("Stop timeout")

    print("==== Client Stopped! ====")

