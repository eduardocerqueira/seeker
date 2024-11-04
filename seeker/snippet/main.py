#date: 2024-11-04T16:50:35Z
#url: https://api.github.com/gists/712ceeb1c4424bc98af1055d15cded2f
#owner: https://api.github.com/users/GGontijo

import json
from pika import BlockingConnection, ConnectionParameters, PlainCredentials, BasicProperties
import uuid
import time

response = None
# Identificador único para a mensagem enviada
corr_id = str(uuid.uuid4())

def on_response(ch, method, properties, body):
    if corr_id == properties.correlation_id:
        global response
        response = body.decode('utf-8')
        print(f'Resposta recebida: {response}')
        ch.basic_ack(delivery_tag=method.delivery_tag)

rmq_rpa_user = ''
rmq_rpa_password = "**********"

_rmq_credentials = PlainCredentials(
    username=rmq_rpa_user, 
    password= "**********"
)
conn_params = ConnectionParameters(host='srvvm736', credentials=_rmq_credentials)

with BlockingConnection(conn_params) as rmq:
    
    print(f'corr_id enviado: {corr_id}')

    channel = rmq.channel()
    queue_declared = channel.queue_declare(queue='', exclusive=True)
    callback_queue = queue_declared.method.queue
    print(callback_queue)
    channel.basic_consume(
        queue=callback_queue,
        on_message_callback=on_response
    )
    properties = BasicProperties(
        reply_to=callback_queue,
        correlation_id=corr_id
    )
    channel.basic_publish(
        exchange='',
        routing_key='xml_prefetch_queue',
        properties=properties,
        body=json.dumps({"chave_acesso": "000000000"})#, "state": "error", "message": "Chave Inválida"})
    )

    start_time = time.time()
    timeout = 180  # Timeout de 3 minutos

    while response is None:
        rmq.process_data_events(time_limit=1)  # Checa eventos a cada 1 segundo
        if time.time() - start_time > timeout:
            raise TimeoutError("Timeout: Nenhuma resposta recebida dentro do tempo limite.")
