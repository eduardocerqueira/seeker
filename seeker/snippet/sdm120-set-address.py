#date: 2026-03-17T17:45:05Z
#url: https://api.github.com/gists/3aefe29d936f81967e45b6b100369cfb
#owner: https://api.github.com/users/glynhudson

import serial
from pymodbus.client import ModbusSerialClient
import time

# Configuration
PORT = '/dev/ttyAMA0'
BAUD = 9600
ORIGINAL_ADDR = 1
NEW_ADDR = 2

# The raw hex command to change address from 1 to 2
# 01 (Addr) 10 (Func) 00 14 (Reg) 00 02 (Qty) 04 (Bytes) 40 00 00 00 (Data) E6 90 (CRC)
change_addr_payload = bytes.fromhex("01 10 00 14 00 02 04 40 00 00 00 E6 90")

def main():
    # Initialize the Modbus RTU Client
    client = ModbusSerialClient(
        port=PORT,
        baudrate=BAUD,
        parity='N',
        stopbits=1,
        bytesize=8,
        timeout=1
    )

    if client.connect():
        print(f"Connected to {PORT}")
        
        # We use the underlying socket/serial send for raw hex payloads
        print(f"Sending command to change address to {NEW_ADDR}...")
        client.socket.write(change_addr_payload)
        
        # Give the device a moment to process and respond
        time.sleep(0.5)
        
        # Read the expected 8-byte response
        response = client.socket.read(8)
        
        if response:
            print(f"Response (Hex): {response.hex(' ').upper()}")
            
            expected_resp = "01100014000201CC"
            if response.hex().upper() == expected_resp:
                print("Success: Address change confirmed by device.")
            else:
                print("Warning: Response received but it did not match the expected confirmation.")
        else:
            print("No response received. Check wiring or power.")
            
        client.close()
    else:
        print(f"Failed to connect on {PORT}")

if __name__ == "__main__":
    main()