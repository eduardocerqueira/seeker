#date: 2022-02-03T16:59:47Z
#url: https://api.github.com/gists/9cd9bbaf9c5bec68d46187bd67399093
#owner: https://api.github.com/users/Staars

#!/usr/bin/env python3

# Usage:
#   pip3 install bleak
#   python3 get_beacon_key.py
#
# List of PRODUCT_ID:
#   339: For 'YLYK01YL'
#   950: For 'YLKG07YL/YLKG08YL'
#   959: For 'YLYB01YL-BHFRC'
#   1254: For 'YLYK01YL-VENFAN'
#   1678: For 'YLYK01YL-FANCL'
#
# Example:
#   python3 get_beacon_key.py


from bleak import BleakClient
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

import asyncio

import random
import re
import sys

MAC_PATTERN = r"^[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}:[0-9A-F]{2}$"

UUID_SERVICE = "fe95"

UUID_AUTH = '00000001-0000-1000-8000-00805f9b34fb'
UUID_FIRMWARE_VERSION = '00000004-0000-1000-8000-00805f9b34fb'
UUID_AUTH_INIT = '00000010-0000-1000-8000-00805f9b34fb'
UUID_BEACON_KEY = '00000014-0000-1000-8000-00805f9b34fb'


MI_KEY1 = bytes([0x90, 0xCA, 0x85, 0xDE])
MI_KEY2 = bytes([0x92, 0xAB, 0x54, 0xFA])
SUBSCRIBE_TRUE = bytes([0x01, 0x00])

KNOWN_PIDS = [339,950,959,1254,1678]

mi_devices = []
client = None

def reverseMac(mac) -> bytes:
    parts = mac.split(":")
    reversedMac = bytearray()
    leng = len(parts)
    for i in range(1, leng + 1):
        reversedMac.extend(bytearray.fromhex(parts[leng - i]))
    return reversedMac


def mixA(mac, productID) -> bytes:
    return bytes([mac[0], mac[2], mac[5], (productID & 0xff), (productID & 0xff), mac[4], mac[5], mac[1]])


def mixB(mac, productID) -> bytes:
    return bytes([mac[0], mac[2], mac[5], ((productID >> 8) & 0xff), mac[4], mac[0], mac[5], (productID & 0xff)])


def cipherInit(key) -> bytes:
    perm = bytearray()
    for i in range(0, 256):
        perm.extend(bytes([i & 0xff]))
    keyLen = len(key)
    j = 0
    for i in range(0, 256):
        j += perm[i] + key[i % keyLen]
        j = j & 0xff
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def cipherCrypt(input, perm) -> bytes:
    index1 = 0
    index2 = 0
    output = bytearray()
    for i in range(0, len(input)):
        index1 = index1 + 1
        index1 = index1 & 0xff
        index2 += perm[index1]
        index2 = index2 & 0xff
        perm[index1], perm[index2] = perm[index2], perm[index1]
        idx = perm[index1] + perm[index2]
        idx = idx & 0xff
        outputByte = input[i] ^ perm[idx]
        output.extend(bytes([outputByte & 0xff]))
    return output


def cipher(key, input) -> bytes:
    # More information: https://github.com/drndos/mikettle
    perm = cipherInit(key)
    return cipherCrypt(input, perm)


def generateRandomToken() -> bytes:
    token = bytearray()
    for i in range(0, 12):
        token.extend(bytes([random.randint(0, 255)]))
    return token

def auth_callback(sender: int, data: bytearray):
    print("Successful notifiction!")

async def get_beacon_key(mac, product_id):
    reversed_mac = None
    token = generateRandomToken()
    auth_cb_success = False

    # Pairing
    input(f"Activate pairing on your '{mac}' device, then press Enter: ")

    # Connect
    # On the Mac we have to do a scan again, else connection fails (at least on M1, macOS 12.2)
    scanner = BleakScanner()
    peripheral = BleakClient(mac)
    devices = await scanner.discover(service_uuids=['0000fe95-0000-1000-8000-00805f9b34fb'])

    for device in devices:
        if device.address == mac:
            print("Connection in progress...")
            peripheral = BleakClient(address_or_ble_device=device)
            conn = await peripheral.connect()
            print("Successful connection!")
            for dev in mi_devices:
                if dev["addr"] == mac:           
                    reversed_mac = reverseMac(dev["MAC"]) # we need the MAC from the advertisement of the first scan

    # Auth (More information: https://github.com/archaron/docs/blob/master/BLE/ylkg08y.md)
    print("Authentication in progress...")
    # Debug info
    # svcs = await peripheral.get_services()
    # auth_descriptor = None
    # auth_char = None
    # for svc in svcs:
    #     print(svc)
    #     for char in svc.characteristics:
    #         print(char)
    #         for desc in char.descriptors:
    #             print(desc)


    await peripheral.write_gatt_char(UUID_AUTH_INIT, MI_KEY1, True)
    await peripheral.start_notify(UUID_AUTH, auth_callback)
    await peripheral.write_gatt_char(UUID_AUTH, cipher(mixA(reversed_mac, product_id), token), True)
    await asyncio.sleep(0.5)
    await peripheral.write_gatt_char(UUID_AUTH, cipher(token, MI_KEY2), True)
    print("Authentication done.")

    # Read
    firmware_version_value = await peripheral.read_gatt_char(UUID_FIRMWARE_VERSION)
    beacon_key_value = await peripheral.read_gatt_char(UUID_BEACON_KEY)

    #  Output
    beacon_key = cipher(token, beacon_key_value).hex()
    try:
        firmware_version = cipher(token, firmware_version_value).decode() # if we got garbage, this will fail
        print("Authentication successful!")
        print(f"beaconKey: '{beacon_key}'")
        print(f"firmware_version: '{firmware_version}'")
    except:
        print("Authentication failed!")
    exit()

async def connect(address):
    async with BleakClient(address_or_ble_device=mi_devices[0]["addr"]) as client:
        print("Model Number:", client)

def scanner_callback(device: BLEDevice, advertisement_data: AdvertisementData):
    print("Found device with name:",device.name)
    mi_device = {}

    mi_device["name"] = device.name
    mi_device["addr"] = device.address
    mi_device["RSSI"] = device.rssi
    buf = advertisement_data.service_data['0000fe95-0000-1000-8000-00805f9b34fb']
    mi_device["PID"] = int.from_bytes([buf[2],buf[3]],"little")
    mi_device["MAC"] = f"{(buf[10]):x}"+":"+f"{(buf[9]):x}"+":"+f"{(buf[8]):x}"+":"+f"{(buf[7]):x}"+":"+f"{(buf[6]):x}"+":"+f"{(buf[5]):x}"
    mi_device["device"] = device

    for dev in mi_devices:
        if dev["addr"] == mi_device["addr"]:
            return  # do not add known devices
        if mi_device["PID"] not in KNOWN_PIDS:
            return # do not add unsupported devices

    mi_devices.append(mi_device)
    print(mi_device)


async def scanner():
    print("Scanning for Xiaomi devices ...")
    service_uuid = [UUID_SERVICE]
    mi_devices = []
    async with BleakScanner(service_uuids=service_uuid) as scanner:
        scanner.register_detection_callback(scanner_callback)
        await asyncio.sleep(5.0)
        # print(devices)
    print("...done.")
    


def main(argv):
    print("Start scanning. Turn or press the knob of your remote to trigger an advertisement.")
    asyncio.run(scanner()) # we need this to get the MAC on the Mac :)

    for device in mi_devices:
        if device["PID"] in KNOWN_PIDS:
            print("Found supported device.")
            asyncio.run(get_beacon_key(device["addr"], device["PID"]))

    print("No supported devices found.")
    print("Known devices are:")
    print("PRODUCT_ID:")
    print("  339: For 'YLYK01YL'")
    print("  950: For 'YLKG07YL/YLKG08YL'")
    print("  959: For 'YLYB01YL-BHFRC'")
    print("  1254: For 'YLYK01YL-VENFAN'")
    print("  1678: For 'YLYK01YL-FANCL'")
    

if __name__ == '__main__':
    main(sys.argv)
