#date: 2022-07-01T17:13:46Z
#url: https://api.github.com/gists/fa119e250064372dbe95808c2f5f38b1
#owner: https://api.github.com/users/gbzarelli

import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class SnsWrapper:
    def __init__(self, sns_resource):
        self.sns_resource = sns_resource

    def get_topic(self, topic_arn: str):
        topic = list(filter(lambda t: (t.arn == topic_arn), self.sns_resource.list_topics()))[0]
        return topic

    def list_topics(self):
        try:
            topics_iter = self.sns_resource.topics.all()
            logger.info("Got topics.")
        except ClientError:
            logger.exception("Couldn't get topics.")
            raise
        else:
            return topics_iter

    def list_subscriptions(self, topic=None):
        try:
            if topic is None:
                subs_iter = self.sns_resource.subscriptions.all()
            else:
                subs_iter = topic.subscriptions.all()
            logger.info("Got subscriptions.")
        except ClientError:
            logger.exception("Couldn't get subscriptions.")
            raise
        else:
            return subs_iter
