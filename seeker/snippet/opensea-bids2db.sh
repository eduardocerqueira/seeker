#date: 2024-04-18T16:49:11Z
#url: https://api.github.com/gists/79a0534d9ed2767d24decc09539fdc57
#owner: https://api.github.com/users/markizano

#!/usr/bin/env python3
'''
Use case: I have an NFT for sale and I want to know when someone makes an offer on it.
I'm tired of refreshing the page at https://opensea.io/assets/ethereum/0x583f608324bce3569472d750b45ae5892d546a04/128
so I can have a custom dashboard if I wish to know the fluctuations of the offers against this item I'm wathing until sold.

This script will suscribe to updates on OpenSea's API and will insert the offers into a MongoDB database.
In this way, I can track updates withouth having to refresh the page so intently and can build my own price graph.

Documentation:
- Get your own OpenSea API token: "**********"://opensea.io/account/settings?tab=developer
- Learn about OpenSea API Streaming (this is where this API call lives): https://docs.opensea.io/reference/stream-api-overview
- PyMongo Docs: https://pymongo.readthedocs.io/en/stable/api/

    pip3 install opensea-stream pymongo dateparser kizano

Saved this script as `~/bin/bids2db` and executed as such:

    chmod 755 ~/bin/bids2db
    bids2db go-fish

Configuration file saved in ~/.config/opensea/config.yml

Sample config file:

```yaml
    api_key: <your-api-key>
    contract_address: '<contract-address-of-NFT>'
    token_id: "**********"
    mongo_uri: 'mongodb://<mongo-user>:<mongo-pass>@<mongo-host>:<mongo-port>/OpenSea?authSource=admin'
    mongo_dbname: OpenSea
```

'''

import sys
import opensea_sdk as opensea
from pymongo import MongoClient
from dateparser import parse as dateparse
from datetime import datetime

import kizano
kizano.Config.APP_NAME = 'opensea'
log = kizano.getLogger(__name__)

class OpenSeaFisher:
    '''
    Go fishing on OpenSea for offers against a specific NFT.
    Establish a client and make the API call for us.
    Return the API data about the offers against the contract address and token id.
    '''
    _instance = None

    @staticmethod
    def getInstance():
        if OpenSeaFisher._instance is None:
            OpenSeaFisher._instance = OpenSeaFisher()
        return OpenSeaFisher._instance

    def __init__(self) -> None:
        self.config = kizano.getConfig()
        self.contract_address = self.config['contract_address']
        self.token_id = "**********"
        log.info('> Connecting to DB ...')
        self.mongo = MongoClient(self.config['mongo_uri'])
        self.db = self.mongo[self.config['mongo_dbname']]
        if not list( self.db.OpenSeaBids.list_indexes() ):
            # Create an index on order_hash to be a unique item.
            self.db.OpenSeaBids.create_index('order_hash', unique=True)
            self.db.OpenSeaBids.create_index('created_date')
        self.openseastream = None
        log.info('> Connected!')

    def __del__(self):
        if hasattr(self, 'unsubscription') and self.unsubscription:
            self.unsubscription()
        if self.openseastream:
            self.openseastream.disconnect()
        if self.mongo:
            self.mongo.close()

    def start(self):
        log.info('Start listening for events...')
        try:
            log.info('> Connecting to OpenSea ...')
            self.openseastream = opensea.OpenseaStreamClient(self.config['api_key'], opensea.Network.MAINNET)
            log.info('> Connected!')
            log.info('> Subscribing to OpenSea events ...')
            self.unsubscription = self.openseastream.onEvents(
                ['bill-murray-1000-destinations'],
                [ '*' ],
                lambda data: self.payload(data))
            log.info('> Subscribed!')
            self.openseastream.startListening()
        except KeyboardInterrupt:
            log.info('Stopping...')
            self.unsubscription()
            self.openseastream.disconnect()
            self.mongo.close()
            self.unsubscription = self.openseastream = self.mongo = None

    def payload(self, payload: dict):
        '''
        Handle the response from the streaming api by writing the records to DB.
        Craft a decent data structure to insert into the DB.
        '''
        # A condensed version of the event, since their data structure is pretty large.
        log.debug(payload)
        try:
            eventType = payload.get('event', payload.get('event_type', ''))
            if eventType not in ('item_received_offer', 'item_received_bid', 'collection_offer'):
                log.info(f'Ignoring event {payload["event"]}')
                return
            if eventType in ('item_received_offer', 'item_received_bid'):
                log.info(f'Offer received for {payload["payload"]["payload"]["nft_id"]}')
                event = {
                    'collection_slug': payload['payload']['payload']['collection']['slug'],
                    'created_date': dateparse(payload['payload']['payload']['created_date']),
                    'event_timestamp': dateparse(payload['payload']['payload']['event_timestamp']),
                    'expiration_date': dateparse(payload['payload']['payload']['expiration_date']),
                    'name': payload['payload']['payload']['metadata']['name'],
                    'nft_id': payload['payload']['payload']['nft_id'],
                    'permalink': payload['payload']['payload']['permalink'],
                    'order_hash': payload['payload']['payload']['order_hash'],
                    'payment_token': "**********"
                    'offer': payload['payload']['payload']['protocol_data']['parameters']['offer'],
                }
            elif eventType == 'collection_offer':
                log.info(f'Offer received for {payload["payload"]["collection"]["slug"]}')
                if payload['payload']['asset_contract_criteria']['address'] != self.contract_address:
                    log.info(f'Ignoring event for {payload["payload"]["asset_contract_criteria"]["address"]} since it is not ours.')
                    return
                event = {
                    'collection_slug': payload['payload']['collection']['slug'],
                    'created_date': dateparse(payload['payload']['created_date']),
                    'event_timestamp': dateparse(payload['payload']['event_timestamp']),
                    'expiration_date': dateparse(payload['payload']['expiration_date']),
                    'nft_id': "**********"
                    'order_hash': payload['payload']['order_hash'],
                    'payment_token': "**********"
                    'offer': payload['payload']['protocol_data']['parameters']['offer'],
                }
            nft_id = "**********"
            if nft_id != event['nft_id']:
                log.info(f'Ignoring event for {event["nft_id"]} since it is not ours.')
                return
            # Insert the event into the DB. The unique index will ignore duplicates for us.
            self.db.OpenSeaBids.insert_one(event)
        except Exception as e:
            log.error(f'Error: ')
            log.error(e)

    def showOffers(self):
        '''
        Fetch the offers from the DB and print them to the console in a pretty format.
        Filter out expired offers.
        '''
        for offer in self.db.OpenSeaBids.find({'expiration_date': {'$gt': datetime.now()}}).sort('created_date', -1):
            # skip records not containing offer, they don't have all the datar.
            if 'offer' not in offer:
                continue
            amount = "**********"
            price = "**********"
            payment_str = "**********"
            log.info(f'{offer["created_date"]} - {payment_str}')


def main():
    '''
    Entrypoint: "**********"
    If there are any new offers we don't have in DB, insert them.
    '''
    log.info('Welcome!')
    fisher = OpenSeaFisher.getInstance()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'go-fish':
        fisher.start()
    elif cmd == 'show-offers':
        fisher.showOffers()
    del fisher, OpenSeaFisher._instance
    log.info('Goodbye!')
    return 0

if __name__ == '__main__':
    sys.exit(main())
with ${TOKEN_ID}
    If there are any new offers we don't have in DB, insert them.
    '''
    log.info('Welcome!')
    fisher = OpenSeaFisher.getInstance()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'go-fish':
        fisher.start()
    elif cmd == 'show-offers':
        fisher.showOffers()
    del fisher, OpenSeaFisher._instance
    log.info('Goodbye!')
    return 0

if __name__ == '__main__':
    sys.exit(main())
