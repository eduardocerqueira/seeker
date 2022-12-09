#date: 2022-12-09T16:43:49Z
#url: https://api.github.com/gists/83e9a540437a010472865362f77d8892
#owner: https://api.github.com/users/derek-rein

from ib_insync import Contract, Stock, Forex, IB
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
import asyncio
from os.path import isdir, isfile
from os import makedirs
import sqlite3


class Mode:
    TRADES = 'TRADES'
    MIDPOINT = 'MIDPOINT'
    BID = 'BID'
    ASK = 'ASK'
    BID_ASK = 'BID_ASK'


class BarSize:
    sec1 = '1 sec'
    sec5 = '5 secs'
    sec10 = '10 secs'
    sec15 = '15 secs'
    sec30 = '30 secs'
    min1 = '1 min'
    min2 = '2 mins'
    min3 = '3 mins'
    min5 = '5 mins'
    min10 = '10 mins'
    min15 = '15 mins'
    min20 = '20 mins'
    min30 = '30 mins'
    hour1 = '1 hour'
    hour2 = '2 hours'
    hour3 = '3 hours'
    hour4 = '4 hours'
    hour8 = '8 hours'
    day = '1 day'
    week = '1 week'

    @classmethod
    def barSizeFromTimeDelta(self, value: timedelta):
        sec = value.total_seconds()
        if sec < 60:
            interval, what = sec, 'sec'
        elif sec < 3600:
            interval, what = int(sec/60), 'min'
        elif sec < 86400:
            interval, what = int(sec/3600), 'hour'
        elif sec >= 86400 * 7:
            return BarSize.week
        else:
            return BarSize.day

        for key in reversed(list(self.__dict__.keys())):
            if key.startswith(what):
                bsInterval = int(getattr(BarSize, key).split()[0])
                if bsInterval <= interval:
                    return getattr(BarSize, key)

class HistoryRequest:
    def __init__(self, contract: Contract, start: datetime, end: datetime, bar_size: str, mode: str):
        self.contract = contract
        self.start = start
        self.end = end
        self.bar_size = bar_size

        if contract.secType == 'CASH' and mode.upper() == 'TRADES':
            mode = 'MIDPOINT'
        self.mode = mode


class IBDowload(object):
    def __init__(self, dataDir):
        self.verbose = True
        self.data_root = dataDir
        self.max_bar_count = 25000
        self.tasks = Queue()

        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=159)
        self.last_request = datetime(2000, 1, 1)

        if not isdir(dataDir):
            makedirs(dataDir)

        #
        db_file = dataDir+"/data2.db"
        is_new = not isfile(db_file)
        self.conn = sqlite3.connect(db_file)

        if is_new:
            cursor = self.conn.cursor()
            cursor.execute("""CREATE TABLE instrument
                              (   conid bigint primary key,
                                  sectype text,
                                  exchange text,
                                  symbol text
                              )   
                              """)

            cursor.execute("""CREATE TABLE barsize
                            (   id integer primary key autoincrement,
                                name text
                            )   
                            """)
            for name, value in BarSize.__dict__.items():
                if not name.startswith('__') and isinstance(value, str):
                    cursor.execute(f'insert into barsize (name) values ("{value}")')

            cursor.execute("""CREATE TABLE mode
                            (   id integer primary key autoincrement,
                                name text
                            )   
                            """)
            for name, value in Mode.__dict__.items():
                if not name.startswith('__') and isinstance(value, str):
                    cursor.execute(f'insert into mode (name) values ("{value}")')

            cursor.execute("""CREATE TABLE ohlc
                            (   conid bigint,
                                barsize int,
                                mode int,
                                time datetime, 
                                open real, 
                                high real, 
                                low real, 
                                close real, 
                                volume integer,
                                primary key (conid, barsize, mode, time)
                            )
                            """)

            cursor.close()
            self.conn.commit()

        # cache barsize and mode keys
        self.bar_size_keys = {}
        cursor = self.conn.cursor()
        for row in cursor.execute('select id, name from barsize').fetchall():
            self.bar_size_keys[row[1]] = row[0]
        cursor.close()

        self.mode_keys = {}
        cursor = self.conn.cursor()
        for row in cursor.execute('select id, name from mode').fetchall():
            self.mode_keys[row[1]] = row[0]
        cursor.close()

        self.run()

    def wait_until_complete(self):
        self.tasks.put(None)
        loop = asyncio.get_event_loop()
        loop.run_forever()

    def run(self):
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(self.worker(), loop)

    async def worker(self):
        while True:
            if self.tasks.empty():
                await asyncio.sleep(0.2)
                continue
            req = self.tasks.get()
            if req is None:
                break
            try:
                await self.do_download(req)
            except Exception as ex:
                print(str(ex))

            self.tasks.task_done()

        loop = asyncio.get_event_loop()
        loop.stop()
        loop.close()

    def get_bar_size_duration(self, bar_size: str):
        val, interval = bar_size.lower().split(' ', 1)

        if  interval.startswith('sec'):
            return float(val)
        elif interval.startswith('min'):
            return float(val) * 60
        elif interval.startswith('hour'):
            return float(val) * 3600
        elif interval.startswith('day'):
            return float(val) * 86400
        elif interval.startswith('week'):
            return float(val) * 86400 * 7
        elif interval.startswith('month'):
            return float(val) * 86400 * 30


    async def do_download(self, req: HistoryRequest):
        time_diff = req.end - req.start
        bar_duration = self.get_bar_size_duration(req.bar_size)

        projected_bar_count = time_diff.total_seconds() / bar_duration

        if projected_bar_count < self.max_bar_count:
            bars = await self.requestData(req.contract, req.start, req.end, req.bar_size, req.mode)
        else:
            bulk_days = int(self.max_bar_count / (86400 / bar_duration))
            bars = []
            date = req.start
            while date < req.end:
                end = min(date+timedelta(days=bulk_days), req.end)
                tmp_bars = await self.requestData(req.contract, date, end, req.bar_size, req.mode)
                while len(tmp_bars) > 0 and tmp_bars[0].date < date:
                    tmp_bars.pop(0)
                bars = bars + tmp_bars

                date = date + timedelta(days=bulk_days)

        cursor = self.conn.cursor()

        # check instrument exists
        query = f'insert or ignore into instrument (conid, symbol, exchange, sectype) values ' \
            f'({req.contract.conId}, "{req.contract.localSymbol}", ' \
            f'"{req.contract.exchange}", "{req.contract.secType}")'
        cursor.execute(query)

        bar_size_key = self.bar_size_keys[req.bar_size]
        mode_key = self.mode_keys[req.mode]
        for bar in bars:
            query = f'insert or replace into ohlc (conid, barsize, mode, time, ' \
                    f'open, high, low, close, volume) values '\
                    f'({req.contract.conId}, "{bar_size_key}", "{mode_key}", ' \
                    f'"{bar.date}", {bar.open}, {bar.high}, {bar.low}, {bar.close}, {bar.volume})'
            cursor.execute(query)
        cursor.close()
        self.conn.commit()

    def qualifyContract(self, contract):
        self.ib.qualifyContracts(contract)
        return contract

    def download(self, contract: Contract, start: datetime, end: datetime, bar_size: str, mode: str):
        """
        :param contract: Contract object contains, conId, symbol, and secType already
        :param start:
        :param end:
        :return:
        """

        # validate args, pass to queue object
        # self.ib.qualifyContracts(contract)
        if contract.conId > 0:
            self.tasks.put(HistoryRequest(contract, start, end, bar_size, mode))
        else:
            raise ValueError("conid is zero. Qualify contract before request data download")

    async def requestData(self, contract, start, end, bar_size, mode):
        """
        Data
        :return:
        """
        query_time = end.strftime("%Y%m%d %H:%M:%S")


        time_diff = end - start
        if time_diff.days >= 365:
            extra_year = 1 if time_diff.days % 365 > 0 else 0
            duration = str(int(time_diff.days / 365) + extra_year) + ' Y'
        elif time_diff.days > 0:
            extra_day = 1 if int(time_diff.total_seconds()) % 86400 > 0 else 0
            duration = str(time_diff.days + extra_day) + ' D'
        else:
            duration = str(int(time_diff.total_seconds())) + ' S'

        if self.verbose:
            print(f'downloading {bar_size} {contract.localSymbol} {duration} before {end}')
        bars = await self.ib.reqHistoricalDataAsync(contract, endDateTime=query_time,
                                                    durationStr=duration,
                                                    barSizeSetting=bar_size, whatToShow=mode,
                                                    useRTH=True)

        if self.verbose:
            print(f'got {len(bars)} bars')

        delay = 10 - (datetime.now() - self.last_request).total_seconds()
        self.last_request = datetime.now()
        if delay > 0:
            if self.verbose:
                print(f'pacing limitations {delay} seconds delay')
            sleep(delay)

        return bars


if __name__ == '__main__':
    downloader = IBDowload('data')

    start = datetime.now()-timedelta(days=10)
    end = datetime.now()-timedelta(days=2)
    bar_size = BarSize.min1
    mode = Mode.TRADES
    items = [(downloader.qualifyContract(Stock('AMD', exchange='SMART', currency='USD')), start, end, bar_size, mode),
             (downloader.qualifyContract(Forex('GBPUSD')), start, end, bar_size, mode),
             (downloader.qualifyContract(Stock('SPY', exchange='SMART', currency='USD')), start, end, bar_size, mode)]

    for i in items:
        downloader.download(*i)

    downloader.wait_until_complete()
