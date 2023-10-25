#date: 2023-10-25T16:43:42Z
#url: https://api.github.com/gists/22c6a2ab28a3a5e85ce6de06857d9870
#owner: https://api.github.com/users/thedolphin

#!venv/bin/python3

import time
import datetime
import logging
import logging.handlers
import threading

import clickhouse_driver

class Timing:

    def __init__(self, logger, context_name):
        self.context_name = context_name
        self.start = datetime.datetime.min
        self.log = logger


    def __enter__(self):
        self.start = datetime.datetime.now()
        self.log.info('begin %s', self.context_name)
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.log.info('finish %s: %s',
            self.context_name,
            datetime.datetime.now() - self.start)


def setup_log(filename):

    log = logging.getLogger()
    log.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    log.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(filename, backupCount=50)
    fh.doRollover()
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    log.addHandler(fh)

    return log


class CHTable:
    def __init__(self, dbconn, dbname, tablename):
        self._dbconn = dbconn
        self._dbname = dbname
        self._tablename = tablename
        self._partitions = None
        self._partition_key = None
        self._partition_key_is_string = None

    @property
    def partitions(self):

        if self._partitions is None:

            res = self._dbconn.execute(
                'select distinct partition '
                'from system.parts '
                'where database = %(dbname)s and table = %(table)s',
                {'dbname': self._dbname, 'table': self._tablename})

            self._partitions = [i[0] for i in res]

        return self._partitions


    @property
    def partition_key(self):

        if self._partition_key is None:

            res = self._dbconn.execute(
                'select partition_key from system.tables '
                'where database = %(database)s and name = %(name)s',
                {'database': self._dbname, 'name': self._tablename})

            self._partition_key = res[0][0]

        return self._partition_key

    @property
    def partition_key_is_string(self):

        if self._partition_key_is_string is None:

            res = self._dbconn.execute(
                f"select {self.partition_key} from {self._dbname}.{self._tablename} limit 1")[0][0]

            self._partition_key_is_string = isinstance(res, (str, datetime.datetime))

        return self._partition_key_is_string


    def _normalize_partition_name(self, partition_name):

        return f"'{partition_name}'" if self.partition_key_is_string else partition_name


    def get_rows_in_partition(self, partition_name):

        normalized_partition_name = self._normalize_partition_name(partition_name)

        res = self._dbconn.execute(
           f'select count(*) from {self._dbname}.{self._tablename} '
           f'where {self.partition_key} = {normalized_partition_name}')

        return res[0][0]


    def start_deduplication_on_partition(self, partition_name):

        res = self._dbconn.execute(
           f'optimize table {self._dbname}.{self._tablename} '
           f'partition {partition_name} deduplicate');


    def get_merges_status(self, partition_name):

        res = self._dbconn.execute(
            'select progress, rows_read, rows_written from system.merges '
            'where database = %(dbname)s and table = %(table)s and partition_id = %(partition)s',
            {'dbname': self._dbname, 'table': self._tablename, 'partition': partition_name})

        return res


def background_deduplication(table, partition):

    table.start_deduplication_on_partition(partition)


def main():

    dbconn = {
        'host': '127.0.0.1',
        'user': 'default',
        'password': "**********"
    }

    dbname = 'public'
    table = 'table'

    db = clickhouse_driver.Client(**dbconn)
    tbl = CHTable(db, dbname, table)

    db2 = clickhouse_driver.Client(**dbconn)
    tbl2 = CHTable(db2, dbname, table)

    partitions = tbl.partitions
#    partitions.remove('...')

    for partition in tbl.partitions:

        rows_before = tbl.get_rows_in_partition(partition)

        with Timing(log, f'partition {partition}'):

            log.info("partition %s: %s rows", partition, rows_before)

            thread = threading.Thread(target=background_deduplication, args=(tbl2, partition))
            thread.start()

            res = True

            while res:
                time.sleep(1)
                res = tbl.get_merges_status(partition)
                if res:
                    log.info('merge status: %s%% done, %s rows read, %s rows written',
                        int(res[0][0] * 100), res[0][1], res[0][2])

            log.info('waiting for query to finish')
            thread.join()


        rows_after = tbl.get_rows_in_partition(partition)
        log.info('partition %s: %s rows before, %s rows after, %s(%s%%) rows deleted',
            partition, rows_before, rows_after,
            rows_before - rows_after,
            int((rows_before - rows_after) * 100/rows_before))

if __name__ == '__main__':

    log = setup_log('optimize.log')
    with Timing(log, 'main'):
        main()
