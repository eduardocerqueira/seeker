#date: 2022-12-22T17:06:47Z
#url: https://api.github.com/gists/0d0919c423758fb55450f43e49740a15
#owner: https://api.github.com/users/1440000bytes

import json
import ssl
import time
import threading
from nostr.filter import Filter, Filters
from nostr.event import Event, EventKind
from nostr.relay_manager import RelayManager
from nostr.message_type import ClientMessageType

relays = set()

def getrelays():

    filters = Filters([Filter(kinds=[2])])
    subscription_id = "crawler"
    request = [ClientMessageType.REQUEST, subscription_id]
    request.extend(filters.to_json_array())

    relay_manager = RelayManager()
    relay_manager.add_relay("wss://nostr-pub.wellorder.net")
    relay_manager.add_subscription(subscription_id, filters)
    relay_manager.open_connections({"cert_reqs": ssl.CERT_NONE})
    time.sleep(1.25)

    message = json.dumps(request)
    relay_manager.publish_message(message)
    time.sleep(1)

    while relay_manager.message_pool.has_events():
        event_msg = relay_manager.message_pool.get_event()        
  
    relay_manager.close_connections()

    return relays.add(event_msg.event.content)

if __name__=="__main__":

    while True:
        getrelays()
        time.sleep(5)
        with open('relays.txt', 'w') as f:
            for relay in relays:
                f.write(relay + '\n')
        thread = threading.Timer(300, getrelays)
        thread.daemon = True
        thread.start()
