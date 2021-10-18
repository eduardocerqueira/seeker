#date: 2021-10-18T17:03:23Z
#url: https://api.github.com/gists/a1b5b43b553a090f075bf444d33db864
#owner: https://api.github.com/users/agtoledo

import asyncio
from random import randint, random
from collections import defaultdict, namedtuple


ShippingLabel = namedtuple("ShippingLabel", "id carrier")


class TrackingJob:
    def __init__(self):
        # Select Shipping labels to track following some strategy
        shipping_labels = get_shipping_labels()
        process_shipping_labels(shipping_labels)


class TaskGroup:
    @classmethod
    async def new(self, tasks):
        await asyncio.gather(*tasks)


def process_shipping_labels(shipping_labels):
    tracker = ShippingLabelTracker(shipping_labels)
    tracker.run()


def get_shipping_labels():
    shipping_labels = []
    for i in range(FAKE_LABELS_PER_CARRIER):
        shipping_labels.append(
            ShippingLabel(carrier="fedex", id=i),
        )
        shipping_labels.append(
            ShippingLabel(carrier="dhl", id=i),
        )

    return shipping_labels


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


async def noise():
    while True:
        await asyncio.sleep(random())
        print("ruido")


class ShippingLabelTracker:
    def __init__(self, shipping_labels):
        self._shipping_labels = shipping_labels

    def run(self):
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                self._process_shipping_labels(self._shipping_labels)
            )

        except KeyboardInterrupt:
            pass

        finally:
            loop.close()

    async def _process_shipping_labels(self, shipping_labels):
        sh_by_carrier = defaultdict(list)
        for sh in shipping_labels:
            sh_by_carrier[sh.carrier].append(sh)

        carrier_tracking_tasks = []
        for carrier, shipping_labels in sh_by_carrier.items():
            carrier_tracker = self._get_carrier_tracker(carrier)
            new_carrier_task = asyncio.create_task(
                carrier_tracker.track_shipping_labels(shipping_labels)
            )
            carrier_tracking_tasks.append(new_carrier_task)

        await TaskGroup.new(carrier_tracking_tasks)

    def _get_carrier_tracker(self, carrier):
        return {
            "fedex": CarrierTracker("FedEx", Client(limit=30), Writer()),
            "dhl": CarrierTracker("DHL", Client(limit=60), Writer()),
        }[carrier]


class CarrierTracker:
    def __init__(self, name, api_client, writer):
        self._name = name
        self._api_client = api_client
        self._writer = writer
        self._max_batches_in_parallalel = asyncio.Semaphore(MAX_BATCHES_IN_PARALLEL)

    async def track_shipping_labels(self, shipping_labels):
        batch_trackers = []
        for batch in self.split_in_batches(shipping_labels):
            batch_trackers.append(
                asyncio.create_task(self.process_batch(batch))
            )
        await TaskGroup.new(batch_trackers)

    async def process_batch(self, batch):
        # TODO Ver si esto puede ser generico para todos los carriers
        batch_input = (
            ShippingLabel(id=sh.id, carrier=sh.carrier)
            for sh in batch
        )
        async with self._max_batches_in_parallalel:
            print(f"Trigger new batch {self._name}: {[i.id for i in batch]}")
            tracking_result = await self._api_client.get_tracking_info(batch_input)
            # TODO Ver q onda el output en dataclass
            await self._writer.process_tracking_result(tracking_result)

    def split_in_batches(self, shipping_labels):
        for chunk in chunks(shipping_labels, self._api_client.batch_limit):
            yield chunk


class Client:
    def __init__(self, limit=1):
        self.batch_limit = limit

    async def get_tracking_info(self, batch):
        await asyncio.sleep(randint(1, 6))
        return f"--> batch result {list(batch)}"


class Writer:
    async def process_tracking_result(self, tracking_result):
        await asyncio.sleep(random())
        print(tracking_result)


MAX_BATCHES_IN_PARALLEL = 3
FAKE_LABELS_PER_CARRIER = 15

TrackingJob()