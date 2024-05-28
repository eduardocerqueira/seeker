#date: 2024-05-28T16:52:39Z
#url: https://api.github.com/gists/57c29d01ccb4a22fe34290d6d0d49731
#owner: https://api.github.com/users/atemate

from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
import os
from time import perf_counter
from contextlib import contextmanager
from typing import Callable
from uuid import uuid4

PROJECT_ID = "foodsci-img-gen-dev-1407-1448"
TOPIC_ID = "artem-aws-manual"
TIMEOUT = None # none means no timeout

SUBSCRIPTION_ID = f"{TOPIC_ID}-sub-view"


subscriber = pubsub_v1.SubscriberClient()
publisher = pubsub_v1.PublisherClient()

topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
# The `subscription_path` method creates a fully qualified identifier in the form `projects/{project_id}/subscriptions/{subscription_id}`
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    print(f"Received (not ack!) {message}")


with subscriber:
    subscription = subscriber.create_subscription(
        request={"name": subscription_path, "topic": topic_path}
    )

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)

    try:
        # When `timeout` is not set, result() will block indefinitely,
        # unless an exception is encountered first.
        streaming_pull_future.result(timeout=TIMEOUT)
    except TimeoutError:
        print(f"timeout {TIMEOUT} sec")
        streaming_pull_future.cancel()  # Trigger the shutdown.
        streaming_pull_future.result()  # Block until the shutdown is complete.
