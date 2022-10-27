#date: 2022-10-27T17:18:41Z
#url: https://api.github.com/gists/0ba1a20b010d0320780f4df3dbbffa14
#owner: https://api.github.com/users/bschne

#!/opt/homebrew/bin/python3
import sys
import asyncio
import platform

from bleak import BleakScanner

ADDRESS = "993192E4-7572-D322-B8C8-589626F55B6D"
CHAR_UUID = "00007001-B38D-4985-720E-0F993A68EE41"

def log_data(data: bytearray):
    byte_order = "little"
    raw = list(data.values())[0]
    data_str = ' '.join('{:02x}'.format(x) for x in raw)

    t_ticks = int.from_bytes(raw[4:6], byte_order, signed=False)
    t = -45 + ((175 * t_ticks) / (2**16-1))

    rh_ticks = int.from_bytes(raw[6:8],  byte_order, signed=False)
    rh = (100 * rh_ticks) / (2**16-1)

    co2 = int.from_bytes(raw[8:10], byte_order, signed=False)

    print(f"{data_str}\t{t:.3f}C\t\t{rh:.3f}%\t\t{co2}ppm")


def detection_callback(device, advertisement_data):
    if device.address == ADDRESS:
        log_data(advertisement_data.manufacturer_data)


async def main():
    stop_event = asyncio.Event()

    print("Listening for sensor data from address:")
    print(ADDRESS)
    print("")
    print("Raw data:\t\t\tTemp.:\t\tRH:\t\tCO2:")

    async with BleakScanner(detection_callback) as scanner:
        await stop_event.wait()


if __name__ == "__main__":
    asyncio.run(main())
