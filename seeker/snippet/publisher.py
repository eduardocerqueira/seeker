#date: 2024-12-26T16:40:25Z
#url: https://api.github.com/gists/537e649d96c69ae8afdaf9b312b45412
#owner: https://api.github.com/users/paulwinex

import asyncio
from random import random
from aiokafka import AIOKafkaProducer


high_prio_queue = 'high-priority-queue'
low_prio_queue = 'low-priority-queue'
server_url = 'localhost:9092'


async def send_message(topic: str, message: str):
    producer = AIOKafkaProducer(bootstrap_servers=server_url)
    await producer.start()
    try:
        await producer.send_and_wait(topic, message.encode('utf-8'))
        print(f"Message sent to topic '{topic}': {message}")
    finally:
        await producer.stop()

async def main():
    for i in range(10):
        if random() < 0.5:
            await send_message(low_prio_queue, "low priority")
        else:
            await send_message(high_prio_queue, "HI PRIORITY!!!")


if __name__ == '__main__':
    asyncio.run(main())
