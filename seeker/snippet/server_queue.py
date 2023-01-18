#date: 2023-01-18T17:04:44Z
#url: https://api.github.com/gists/eb5dd744040b641654ab339e3c544c94
#owner: https://api.github.com/users/andrequeiroz2

from pika import BlockingConnection, ConnectionParameters, BasicProperties

connection = BlockingConnection(
    ConnectionParameters(host='localhost', port=5672)
)

channel = connection.channel()

channel.queue_declare(queue='queue_user')


def queue_user_callback(ch, method, props, body):

    data = body
    response = 'ok'

    ch.basic_publish(
        exchange='',
        routing_key=props.reply_to,
        properties=BasicProperties(correlation_id=props.correlation_id),
        body=str(response)
    )

    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)

channel.basic_consume(queue='queue_user', on_message_callback=queue_user_callback)

print("start")
channel.start_consuming()