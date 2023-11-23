#date: 2023-11-23T16:43:20Z
#url: https://api.github.com/gists/b61328593df9a532c91f3d0ed9ed7541
#owner: https://api.github.com/users/Sygmei

import pika

credentials = "**********"="guest", password="guest")
connection = pika.BlockingConnection(
    pika.ConnectionParameters("188.166.21.104", 5672, "test_vhost", credentials)
)
channel = connection.channel()
countmessage = 1
messages = "Hello, A message to Rabbit MQ by Python Script"
for x in range(countmessage):
    messages = messages.format(x)
    channel.basic_publish(
        exchange="#exchange_name",
        routing_key="#routingkey",
        body=messages,
        properties=pika.BasicProperties(delivery_mode=2),
    )
    print("Sent Message no {}".format(x))
connection.close()
print("Sending Done")
t("Sending Done")
