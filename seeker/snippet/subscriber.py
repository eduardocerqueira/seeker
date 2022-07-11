#date: 2022-07-11T16:53:13Z
#url: https://api.github.com/gists/db0737de212bb5286e9d744ee8168cc5
#owner: https://api.github.com/users/caitray13

from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
import json

from model import ModelPrediction
import publisher

SUBSCRIPTION_ID = "app-sub"
TIMEOUT = 60.0

def sub(project_id, subscription_id, timeout):
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)

    
    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        print(f"Received {message}.")
        message.ack()
        # extract message data
        message_data = message.data.decode()
        message_json = json.loads(message_data)
        image_bytes = str.encode((message_json["image"]))
        # model prediction
        modelPrediction = ModelPrediction()
        label = str(modelPrediction.predict(image_bytes))
        print(f"Label is {label}")
        # submit label to publisher
        publisher.pub(label)

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening for messages on {subscription_path}..\n")

    with subscriber:
        try:
            # When `timeout` is not set, result() will block indefinitely,
            # unless an exception is encountered first.
            streaming_pull_future.result(timeout=timeout)
        except TimeoutError:
            streaming_pull_future.cancel()
            streaming_pull_future.result()

if __name__=='__main__':
    print('Starting subscriber...')
    sub(PROJECT_ID, SUBSCRIPTION_ID, TIMEOUT)