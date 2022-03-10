#date: 2022-03-10T16:58:17Z
#url: https://api.github.com/gists/7439889747e5a89e9bfcc63d2b3ca48f
#owner: https://api.github.com/users/waterquarks

import csv
import json
import os
import time
from threading import Event, Thread
from time import time

import mango
from mango.context import Context
from mango.perpeventqueue import PerpFillEvent
from mango.perpmarket import PerpMarket

from lib.extended_encoder import ExtendedEncoder


def subscribe_to_trades(context: Context, market: PerpMarket):
    file_path = f"./trades/{market.symbol}.csv"

    fieldnames = [
        'exchange',
        'symbol',
        'timestamp',
        'taker',
        'taker_order',
        'taker_client_order_id',
        'maker',
        'maker_order',
        'maker_client_order_id',
        'side',
        'price',
        'amount'
    ]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as file:
        writer = csv.DictWriter(file, fieldnames)

        writer.writeheader()

        file.flush()

        def store(fill: PerpFillEvent):
            output = {
                'exchange': 'Mango Markets',
                'symbol': market.symbol,
                'timestamp': int(time() * 1e6),
                'taker': fill.taker,
                'taker_order': fill.taker_order_id,
                'taker_client_order_id': fill.taker_client_order_id,
                'maker': fill.maker,
                'maker_order': fill.maker_order_id,
                'maker_client_order_id': fill.maker_client_order_id,
                'side': fill.taker_side,
                'price': fill.price,
                'amount': fill.quantity
            }

            print(json.dumps(output, cls=ExtendedEncoder))

            writer.writerow(output)

            file.flush()

        market.on_fill(context, store)

        waiter = Event()

        try:
            waiter.wait()
        except Exception:
            pass

if __name__ == '__main__':
    with mango.ContextBuilder.build(cluster_name='mainnet') as context:

        markets = [mango.market(context, market) for market in [
            'BTC-PERP',
            'SOL-PERP',
            'MNGO-PERP',
            'ADA-PERP',
            'AVAX-PERP',
            'BNB-PERP',
            'ETH-PERP',
            'FTT-PERP',
            'LUNA-PERP',
            'MNGO-PERP',
            'RAY-PERP',
            'SRM-PERP'
        ]]

        [Thread(target=subscribe_to_trades, args=(context, market)).start() for market in markets]
