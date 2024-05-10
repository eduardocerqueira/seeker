#date: 2024-05-10T16:49:00Z
#url: https://api.github.com/gists/7576be664f74384cae850c5c1c264755
#owner: https://api.github.com/users/hlord2000

# This demo shows how to connect to a device, get its services and characteristics,
# and then subscribe to notifications from a characteristic. It also shows how to
# write to a characteristic.
#
# To begin, create a virtual environment and install bleak
#
# python3 -m venv venv
# source venv/bin/activate
# pip install bleak
#
# Then run python3 main.py

import asyncio
from bleak import BleakClient

# You can change the address to your device's address. Alternatively,
# it is possible to scan for devices and filter by name. For this example,
# I chose to just scan separately and get the device's address.
address = "D5:5A:54:BE:64:F4"

async def notification_handler(sender, data):
    print(f"Notification from {sender}: {data}")

# This is the main function that will be run. It will connect to the device
# it uses Python async/await syntax so looks a bit complicated
async def main(address):
    async with BleakClient(address) as client:
        # If the client is not connected, connect to it
        if not client.is_connected:
            await client.connect()
        
        # Get the services of the device
        services = client.services
        # For each service, print the path, uuid, and characteristics
        nordic_uart_tx = None
        nordic_uart_rx = None
        for service in services:
            print(f"Service Description: {service.description}")
            print(f"Service UUID: {service.uuid}")
            # For each characteristic in the service, print the uuid and description
            for characteristic in service.characteristics:
                if ("Nordic UART TX" in characteristic.description):
                    nordic_uart_tx = characteristic

                if ("Nordic UART RX" in characteristic.description):
                    nordic_uart_rx = characteristic

                print(f"Characteristic UUID: {characteristic.uuid}")
                print(f"Characteristic Description: {characteristic.description}")
        # Now let's subscribe to notifications from the Nordic UART service's TX characteristic
        # Note that we created a "notification_handler" function above that will be called 
        # when a notification is received
        await client.start_notify(nordic_uart_tx.uuid, notification_handler)

        # Now let's write a message to the Nordic UART service's RX characteristic
        await client.write_gatt_char(nordic_uart_rx.uuid, b'Hello World!')

        await asyncio.sleep(30)

asyncio.run(main(address))
