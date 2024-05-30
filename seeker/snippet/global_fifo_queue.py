#date: 2024-05-30T17:01:56Z
#url: https://api.github.com/gists/e995eec06b8bb08d5c5b391fe75ebbf7
#owner: https://api.github.com/users/JacobZuliani

from time import sleep

from google.cloud import pubsub
from google.cloud.pubsub_v1.types import PublisherOptions, Topic, Subscription
from google.protobuf.duration_pb2 import Duration
from google.api_core.exceptions import AlreadyExists, NotFound


PROJECT_ID = "burla-test"


class Empty(Exception):
    pass


class PubSubQueue:
    """
    Global distributed FIFO queue built on google cloud pubsub.
    Only works within the same single gcp region, I think.
    """

    def _init(self, id: str):
        self.id = id
        publisher_options = PublisherOptions(enable_message_ordering=True)
        self.publisher = pubsub.PublisherClient(publisher_options=publisher_options)
        self.subscriber = pubsub.SubscriberClient()
        self.topic_path = self.publisher.topic_path(PROJECT_ID, self.id)
        subscription_id = f"singleton-subscription_for_{self.id}"
        self.subscription_path = self.subscriber.subscription_path(PROJECT_ID, subscription_id)

    def __init__(self, id: str, msg_retention_duration_sec: int = 60 * 60 * 24 * 7):
        self._init(id)
        # create topic
        try:
            topic = Topic(name=self.topic_path, labels={"burla-component": "true"})
            self.publisher.create_topic(request=topic)
        except AlreadyExists:
            msg = f'Queue with id: "{self.id}" already exists, queue-id\'s must be globally unique.'
            raise AlreadyExists(msg)
        # create subscription
        subscription = Subscription(
            name=self.subscription_path,
            topic=self.topic_path,
            ack_deadline_seconds=10,
            retain_acked_messages=False,
            enable_exactly_once_delivery=True,
            enable_message_ordering=True,
            message_retention_duration=Duration(seconds=msg_retention_duration_sec),  # max 7 days
            labels={"burla-component": "true"},
        )
        self.subscriber.create_subscription(request=subscription)

    @classmethod
    def from_id(cls, id: str):
        self = PubSubQueue.__new__(PubSubQueue)
        self._init(id)
        # check if topic for this queue exists.
        # we assume if the topic exists the subsctiption also exists.
        try:
            self.publisher.get_topic(topic=self.topic_path)
        except NotFound:
            raise NotFound(f'Queue with id: "{self.id}" not found.')
        return self

    def put(self, message: str):
        self.publisher.publish(self.topic_path, message.encode("utf-8"), ordering_key=self.id)

    def pop(self) -> str:
        response = self.subscriber.pull(subscription=self.subscription_path, max_messages=1)
        messages = response.received_messages

        if messages:
            ack_id = messages[0].ack_id
            self.subscriber.acknowledge(subscription=self.subscription_path, ack_ids=[ack_id])
            return messages[0].message.data.decode()
        else:
            raise Empty("queue currently empty.")

    def delete(self):
        self.publisher.delete_topic(topic=self.topic_path)
        self.subscriber.delete_subscription(subscription=self.subscription_path)


QUEUE_ID = "test-queue-1"
try:
    queue = PubSubQueue(id=QUEUE_ID)
except AlreadyExists:
    queue = PubSubQueue.from_id(QUEUE_ID)


# queue.delete()


# n_messages = 100
# for x in range(1, n_messages + 1):
#     message = f"test message number {x}"
#     queue.put(message)
#     print(f"Added message to queue: {message}")
#     sleep(0.05)


while True:
    try:
        message = queue.pop()
        print(f"popped message: {message}")
    except Empty:
        print("queue is empty")
        break
