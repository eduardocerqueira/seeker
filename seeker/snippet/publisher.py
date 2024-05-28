#date: 2024-05-28T16:52:39Z
#url: https://api.github.com/gists/57c29d01ccb4a22fe34290d6d0d49731
#owner: https://api.github.com/users/atemate

from google.cloud import pubsub_v1
from uuid import uuid4


# TODO(developer)
project_id = "..."  # dev proj id
topic_id = "artem-aws-manual"

publisher = pubsub_v1.PublisherClient()
# The `topic_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/topics/{topic_id}`
topic_path = publisher.topic_path(project_id, topic_id)

for n in range(1, 10):
    data_str = f"Message: {uuid4()}"
    # Data must be a bytestring
    data = data_str.encode("utf-8")
    # When you publish a message, the client returns a future.
    attributes = {"request_id": f"request-{n}"}
    future = publisher.publish(topic_path, data, **attributes)
    print(future.result())

print(f"Published messages to {topic_path}.")
