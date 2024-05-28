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

N = os.environ["N"]  # emulates the pipeline - will receive "request-N"
PROJECT_ID = "..."  # dev proj id
TOPIC_ID = "artem-aws-manual"
TIMEOUT = None # none means no timeout

vertex_pipeline_id = uuid4()  # here for uniqueness
SUBSCRIPTION_ID = f"{TOPIC_ID}-sub-{vertex_pipeline_id}-{N}"



@contextmanager
def catchtime() -> Callable[[], float]:
    t1 = t2 = perf_counter() 
    yield lambda: t2 - t1
    t2 = perf_counter() 


subscriber = pubsub_v1.SubscriberClient()
publisher = pubsub_v1.PublisherClient()

topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
# The `subscription_path` method creates a fully qualified identifier in the form `projects/{project_id}/subscriptions/{subscription_id}`
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

filter = f'attributes.request_id="request-{N}"'

def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    print(f"Received {message}")
    message.ack()


with subscriber:
    try:
        with catchtime() as t:
            subscription = subscriber.create_subscription(
                request={"name": subscription_path, "topic": topic_path, "filter": filter}
            )
        print(f"Created subscription within {t():.3f} sec: {subscription}")

        with catchtime() as t:
            streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
        print(f"Subscribed within {t():.3f} sec")

        try:
            # When `timeout` is not set, result() will block indefinitely,
            # unless an exception is encountered first.
            streaming_pull_future.result(timeout=TIMEOUT)
        except TimeoutError:
            print(f"timeout {TIMEOUT} sec")
            streaming_pull_future.cancel()  # Trigger the shutdown.
            streaming_pull_future.result()  # Block until the shutdown is complete.

    finally:
        with catchtime() as t:
            subscriber.delete_subscription(request={"subscription": subscription_path})
        print(f"Subscription deleted within {t():.3f} sec")

