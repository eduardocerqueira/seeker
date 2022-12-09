#date: 2022-12-09T16:43:49Z
#url: https://api.github.com/gists/83e9a540437a010472865362f77d8892
#owner: https://api.github.com/users/derek-rein

from ib_insync import Contract, Stock, Forex
from datetime import datetime, timedelta
from threading import Lock
import pandas as pd
from ibDownloader.downloader import IBDowload, Mode, BarSize

class DataBase(object):
    """
    For accessing data that has been downloaded and saved already
    Making requests to the downloader to extend alredy existing data
    """
    def __init__(self, dataDir):
        self._dataDir = dataDir
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.extend_lock = Lock()
        self._downloader = IBDowload(dataDir)

    def validateBarSize(self, value: str) -> bool:
        for key, val in BarSize.__dict__.items():
            if val == value:
                return True

        return False

    def getContract(self, contract):
        return self._downloader.qualifyContract(contract)

    def getData(self, contract: Contract, barSize: BarSize, mode: Mode, start: datetime=None, end: datetime=None):
        """
        Given contract and barsize, start/end==None get the whole dataframe
        With a start, get start to the last available
        With end, get first available to the end
        With start and end ...
        """
        query = f'select time, open, high, low, volume from ohlc where ' \
                f'conid={contract.conId} and barsize="{self._downloader.bar_size_keys[barSize]}" ' \
                f'and mode="{self._downloader.mode_keys[mode]}"'
        if start is not None:
            query +=f' and time >= "{start.strftime(self.date_format)}"'
        if end is not None:
            query += f' and time < "{end.strftime(self.date_format)}"'

        query += " order by time"

        df = pd.read_sql(query, self._downloader.conn, parse_dates='time', index_col='time')
        return df

    def dict(self):
        """
        Returns a tree-like dictionary following the storage schema
        /forex/idealpro/gbpusd/5mins/midpoints/gbpusd.db
        """

        d = {}
        query = 'select distinct i.sectype, i.exchange, i.symbol, b.name, m.name ' \
                'from ohlc o left join instrument i on o.conid=i.conid ' \
                'left join barsize b on o.barsize = b.id ' \
                'left join mode m on o.mode = m.id'
        cursor = self._downloader.conn.cursor()
        cursor.execute(query)
        for row in cursor.fetchall():
            if row[0] not in d:
                d[row[0]] = {}
            if row[1] not in d[row[0]]:
                d[row[0]][row[1]] = {}
            if row[2] not in d[row[0]][row[1]]:
                d[row[0]][row[1]][row[2]] = {}

            d[row[0]][row[1]][row[2]][row[3]] = row[4]

        cursor.close()
        return d

    def extendData(self, contract, barSize, mode, requested_start, requested_end):
        """
        Extends data for a given contract/barsize to the current time
        Does a booleon  check on existing data vs requested data,
        sets correct start and end times to avoid downloading the same data twice
        if the request start/end date range is larger than whats in the existing data; extend the existing data backwards and forwards
        """
        with self.extend_lock:
            existing_start, existing_end = self.getDateRange(contract, barSize, mode)
            if existing_start is None:
                self._downloader.download(contract, requested_start, requested_end, barSize, mode)
                return

            if requested_start < existing_start:
                self._downloader.download(contract, requested_start, existing_start, barSize, mode)
            if requested_end > existing_end:
                self._downloader.download(contract, existing_end + timedelta(seconds=1),
                                          requested_end, barSize, mode)

    def extendAllDataToCurrentTime(self):
        """
        Extends all existing data that has been downloaded to current time
        """
        data = self.dict()
        for secType, exchanges in data.items():
            for exchange, symbols in exchanges.items():
                for symbol, barSizes in symbols.items():
                    if secType == 'cash':
                        contract = Forex(symbol.replace('.', ''))
                    elif secType == 'stk':
                        contract = Stock(symbol, exchange=exchange, currency='USD')
                    else:
                        contract = Contract()
                        contract.secType = secType.upper()
                        contract.exchange = exchange.upper()
                        contract.localSymbol = symbol.upper()
                        contract.currency = 'USD'

                    contract = self.getContract(contract)

                    for barSize, mode in barSizes.items():
                        end_of_existing_data = self.getDateRange(contract, barSize, mode)[1]
                        self._downloader.download(contract,
                                                  end_of_existing_data + timedelta(seconds=1),
                                                  datetime.now(), barSize, mode)

    def getDateRange(self, contract, barSize, mode):
        """
        Gets a tuple datetime objects for the first, last available bar given a contract and barsize
        """
        try:
            cursor = self._downloader.conn.cursor()
            query = f'select min(time), max(time) from ohlc where ' \
                    f'conid={contract.conId} and ' \
                    f'barsize="{self._downloader.bar_size_keys[barSize]}" and ' \
                    f'mode="{self._downloader.mode_keys[mode]}"'
            cursor.execute(query)
            row = cursor.fetchall()
            if row is None or len(row) != 1:
                return None, None

            return datetime.strptime(row[0][0], self.date_format), \
                   datetime.strptime(row[0][1], self.date_format)
        except KeyError:
            return None, None

if __name__ == '__main__':
    db = DataBase('data')
    if db.validateBarSize('1 min'):
        print('Bar size is valid')

    print(BarSize.barSizeFromTimeDelta(timedelta(seconds=5)))
    print(BarSize.barSizeFromTimeDelta(timedelta(seconds=17)))
    print(BarSize.barSizeFromTimeDelta(timedelta(minutes=21)))
    print(BarSize.barSizeFromTimeDelta(timedelta(hours=2)))
    print(BarSize.barSizeFromTimeDelta(timedelta(days=3)))
    print(BarSize.barSizeFromTimeDelta(timedelta(days=7)))
    print(BarSize.barSizeFromTimeDelta(timedelta(days=45)))
    #exit(0)

    print('dict:', db.dict())

    contract = db.getContract(Stock('AMD', exchange='SMART', currency='USD'))
    #contract = db.getContract(Forex('GBPUSD'))

    begin, end = db.getDateRange(contract, BarSize.min1, Mode.TRADES)
    print(f'{contract.localSymbol} data from {begin} to {end}')

    print(f'head of {contract.localSymbol} {BarSize.min1} data:')
    df = db.getData(contract, BarSize.min1, Mode.TRADES)
    print(df.head())

    df = db.getData(contract, BarSize.min1, Mode.TRADES, datetime(2019, 12, 12))
    print(df.head())

    df = db.getData(contract, BarSize.min1, Mode.TRADES,
                    datetime(2019, 12, 12, 9, 45, 0), datetime(2019, 12, 12, 10, 0, 0))
    print(df.head())

    print('Extending all data')
    db.extendAllDataToCurrentTime()

    print('extend AMD data')
    db.extendData(contract, BarSize.min1, Mode.TRADES,
                  datetime.now() - timedelta(days=24), datetime.now())

    print('waiting all tasks complete')
    db._downloader.wait_until_complete()
