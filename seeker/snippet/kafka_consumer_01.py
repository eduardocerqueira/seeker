#date: 2022-10-27T17:20:34Z
#url: https://api.github.com/gists/2241debeda34cff01e0bcc5db0a44a81
#owner: https://api.github.com/users/amanjaiswalofficial

from confluent_kafka import Producer

class KafkaProducer:
    def __init__(self, **kwargs: Dict) -> None:
        self.topic = kwargs.pop("topic")
        self.partition = kwargs.pop("partition")
        self.producer = Producer(**kwargs)
        self.on_delivery = kwargs.get("on_delivery_ref")
        self.enable_flush = kwargs.get("enable_flush", False)