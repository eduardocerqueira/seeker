#date: 2023-01-04T16:55:39Z
#url: https://api.github.com/gists/a00dd3ccf0635a05b239f75136ce19ad
#owner: https://api.github.com/users/odudex

""" Python nostr client for air-gapped shit posting
    using https://github.com/jeffthibault/python-nostr
"""

from io import StringIO
from python_nostr.nostr.event import Event
from python_nostr.nostr.relay_manager import RelayManager
from python_nostr.nostr.message_type import ClientMessageType

# import the opencv library
import cv2

import json 
import ssl
import time

from embit import bech32
from qrcode import QRCode

def scan():
    vid = cv2.VideoCapture(0)
    detector = cv2.QRCodeDetector()
    qr_data = None
    while True:
        # Capture the video frame by frame
        ret, frame = vid.read()
        qr_data, bbox, straight_qrcode = detector.detectAndDecode(frame)
        if len(qr_data) > 0:
            break
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    return qr_data

# Connecting to relays
print("Connecting to relays")
relay_manager = RelayManager()
relay_manager.add_relay("wss://nostr-pub.wellorder.net")
relay_manager.add_relay("wss://relay.damus.io")
relay_manager.open_connections({"cert_reqs": ssl.CERT_NONE}) # NOTE: This disables ssl certificate verification
time.sleep(1.25) # allow the connections to open


# Scanning public keys
print("Scanning public key ...")
pubkey_qr_data = scan()
print("bech32 key: " + pubkey_qr_data)

# Convert bech32 key to hex
spec, hrp, data = bech32.bech32_decode(pubkey_qr_data)
hex_public_key = bytes(bech32.convertbits(data, 5, 8, False)).hex()
print("Hex Key: " + hex_public_key)

# Create message:
message = input("Type your message: ")

# Create event
event = Event(hex_public_key, message)

# Create event ID QR code in 32 bytes format
event_id_qr = QRCode()
event_id_qr.add_data(bytes.fromhex(event.id))
qr_string = StringIO()
event_id_qr.print_ascii(out=qr_string, invert=True)

print("Event ID: " + str(event.id))
print("Scan it with your signing device")
print(qr_string.getvalue())

_ = input("Press enter to scan signed event: ")
event_id_signature = scan()
print("signature: " + event_id_signature)
event.signature = event_id_signature

message = json.dumps([ClientMessageType.EVENT, event.to_json_object()])
relay_manager.publish_message(message)
time.sleep(1) # allow the messages to send

relay_manager.close_connections()
