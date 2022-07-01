#date: 2022-07-01T17:13:46Z
#url: https://api.github.com/gists/fa119e250064372dbe95808c2f5f38b1
#owner: https://api.github.com/users/gbzarelli

from sns_wrapper import SnsWrapper

import boto3

ACCESS_KEY = "-"
SECRET_KEY = "-"

REGION_FROM = "sa-east-1"
TOPIC_ARN_FROM = "arn:aws:sns:sa-east-1:000000000:teste-zarelli-topic"

REGION_TO = "us-east-1"
TOPIC_ARN_TO = "arn:aws:sns:us-east-1:000000000:teste-zarelli-topic"


if __name__ == '__main__':
    session = boto3.Session(
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY)

    # Load resources
    resource_from = session.resource(service_name='sns', region_name=REGION_FROM)
    resource_to = session.resource(service_name='sns', region_name=REGION_TO)
    sns_wrapper_from = SnsWrapper(resource_from)
    sns_wrapper_to = SnsWrapper(resource_to)

    # Load topics
    topic_from = sns_wrapper_from.get_topic(TOPIC_ARN_FROM)
    topic_to = sns_wrapper_to.get_topic(TOPIC_ARN_TO)

    subscriptions_from = sns_wrapper_from.list_subscriptions(topic_from)

    # Migrate subs
    for sub in subscriptions_from:
        attrs = {'RawMessageDelivery': sub.attributes['RawMessageDelivery']}
        if sub.attributes.get('FilterPolicy') is not None:
            attrs['FilterPolicy'] = sub.attributes['FilterPolicy']

        new_subscription = topic_to.subscribe(
            Protocol=sub.attributes['Protocol'],
            Endpoint=sub.attributes['Endpoint'],
            Attributes=attrs,
            ReturnSubscriptionArn=True
        )
        print(f'Subscribe to {new_subscription.attributes["Endpoint"]} - Pending?: '
              f'{new_subscription.attributes["PendingConfirmation"]}')

