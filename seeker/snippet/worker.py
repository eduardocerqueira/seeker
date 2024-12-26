#date: 2024-12-26T16:40:25Z
#url: https://api.github.com/gists/537e649d96c69ae8afdaf9b312b45412
#owner: https://api.github.com/users/paulwinex

import asyncio
from aiokafka import AIOKafkaConsumer


high_prio_queue = 'high-priority-queue'
low_prio_queue = 'low-priority-queue'
server_url = 'localhost:9092'


async def process_message(priority: str, message):
    print(f"[{priority} queue] Message: {message}")
    await asyncio.sleep(1)


async def consume():
    high_priority_consumer = AIOKafkaConsumer(
        high_prio_queue,
        bootstrap_servers=server_url,
        group_id="priority_group",
    )

    normal_priority_consumer = AIOKafkaConsumer(
        low_prio_queue,
        bootstrap_servers=server_url,
        group_id="priority_group",
    )

    await high_priority_consumer.start()
    await normal_priority_consumer.start()

    try:
        while True:
            try:
                msg = await asyncio.wait_for(high_priority_consumer.getone(), timeout=0.1)
                if msg:
                    await process_message("High Priority", msg.value.decode('utf-8'))
                    continue
            except asyncio.TimeoutError:
                pass
            try:
                msg = await asyncio.wait_for(normal_priority_consumer.getone(), timeout=0.1)
                if msg:
                    await process_message("Low Priority", msg.value.decode('utf-8'))
                    continue
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        pass
    finally:
        await high_priority_consumer.stop()
        await normal_priority_consumer.stop()

async def main():
    await consume()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped")
