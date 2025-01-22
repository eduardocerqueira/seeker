#date: 2025-01-22T16:57:04Z
#url: https://api.github.com/gists/024f4daa1bc62afa2735b508e91333a0
#owner: https://api.github.com/users/jaredcrean

# Derived from this: https://github.com/gashton/bambustudio_tools/blob/master/bambudiscovery.sh
# Python implementation without need for linux
# Send the IP address of your BambuLab printer to port 2021/udp, which BambuStudio is listens on.

# Ensure your PC has firewall pot 2021/udp open. This is required as the proper response would usually go to the ephemeral source port that the M-SEARCH ssdp:discover message.
# But we are are blindly sending a response directly to the BambuStudio listening service port (2021/udp).

# Temporary solution to BambuStudio not allowing you to manually specify the Printer IP.

# Usage:
# 0. Edit the constants below with your printer SN, model name and the friendly name you want to see in Studio / Orca Slicer
# 1. start Bambu Studio / Orca Slicer
# 2. python bambu-ssdp-discovery.py PRINTER_IP
# 3. connect to the printer

# The script needs to be run every time you start Studio or Orca Slicer

import sys
import socket
from datetime import datetime

TARGET_IP = "127.0.0.1" # Change this to the IP of the computer with the printer software. If you're running this on the same computer, leave it as is.
PRINTER_USN = "YOUR_PRINTER_SN" # This is the serial number of the printer. https://wiki.bambulab.com/en/general/find-sn
PRINTER_DEV_MODEL = "3DPrinter-X1-Carbon" # "3DPrinter-X1-Carbon", "3DPrinter-X1", "C11" (for P1P), "C12" (for P1S), "C13" (for X1E), "N1" (A1 mini), "N2S" (A1)
PRINTER_DEV_NAME = "X1C-1" # The friendly name displayed in Bambu Studio / Orca Slicer. Set this to whatever you want.
PRINTER_DEV_SIGNAL = "-44" # Fake wifi signal strength
PRINTER_DEV_CONNECT = "lan" # printer is in lan only mode
PRINTER_DEV_BIND = "free" # and is not bound to any cloud account
PRINTER_IP = None # If you want to hardcode the printer IP, set it here. Otherwise, pass it as the first argument to the script.
TARGET_PORT = 2021 # The port used for SSDP discovery

def send_udp_response(response):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.sendto(response.encode(), (TARGET_IP, TARGET_PORT))
            print("UDP packet sent successfully.")
        except socket.error as e:
            print("Error sending UDP packet:", e)

def resolve_and_validate(input_str):
    """Resolve a hostname or FQDN to an IP address, or just return the IP address after validating it."""
    try:
        # This will work for both FQDN and hostname
        return socket.gethostbyname(input_str)
    except socket.gaierror:
        # If resolution fails, check if it's a valid IP
        try:
            socket.inet_aton(input_str)
            return input_str  # It's a valid IP, so return it as-is
        except socket.error:
            print(f"Unable to resolve {input_str} to an IP address.")
            sys.exit(2)

def main():
    if PRINTER_IP is None:
        # If PRINTER_IP is not set, check if it was passed as an argument
        if len(sys.argv) == 2:
            provided_ip = sys.argv[1]
        else:
            print("Please specify your printer's IP, FQDN or hostname.\nusage:", sys.argv[0], "<PRINTER_IP>\nAlternatively, set PRINTER_IP in the script.")
            sys.exit(2)
    else:
        # If PRINTER_IP is set, use that
        provided_ip = PRINTER_IP

    # Now that we have a printer IP, FQDN or hostname, resolve and validate it
    printer_ip = resolve_and_validate(provided_ip)

    response = (
        f"HTTP/1.1 200 OK\r\n"
        f"Server: Buildroot/2018.02-rc3 UPnP/1.0 ssdpd/1.8\r\n"
        f"Date: {datetime.now()}\r\n"
        f"Location: {printer_ip}\r\n"
        f"ST: urn:bambulab-com:device:3dprinter:1\r\n"
        f"EXT:\r\n"
        f"USN: {PRINTER_USN}\r\n"
        f"Cache-Control: max-age=1800\r\n"
        f"DevModel.bambu.com: {PRINTER_DEV_MODEL}\r\n"
        f"DevName.bambu.com: {PRINTER_DEV_NAME}\r\n"
        f"DevSignal.bambu.com: {PRINTER_DEV_SIGNAL}\r\n"
        f"DevConnect.bambu.com: {PRINTER_DEV_CONNECT}\r\n"
        f"DevBind.bambu.com: {PRINTER_DEV_BIND}\r\n\r\n"
    )
    print(f"Sending response with PRINTER_IP={printer_ip} to {TARGET_IP}:{TARGET_PORT}")
    send_udp_response(response)

if __name__ == "__main__":
    main()