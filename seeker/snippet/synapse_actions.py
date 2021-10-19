#date: 2021-10-19T17:02:57Z
#url: https://api.github.com/gists/3592614c1d4fb767ccbccc122f82dade
#owner: https://api.github.com/users/Echo9k

import pyodbc
import logging
import argparse
import os
import pathlib
import re


def create_connection(SERVER, DATABASE, UID, PWD):
    connect_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s' % (
        SERVER, DATABASE, UID, PWD)
    conn = pyodbc.connect(connect_str)
    conn.autocommit = True
    return conn


def make_query(query_type, *, bucket, table_name, erpHost, database, **kwargs):
    if query_type == 'count':
        query = f'''
            SELECT count(*)
            FROM openrowset(
                bulk '{bucket}/{table_name}/erpHost={erpHost}/erpDatabase={database}/**',
                data_source = '{kwargs['data_source']}',
                format = '{kwargs['format']}'
            ) AS rows
        '''
        if (kwargs['from_date'] is not None) and (kwargs['to_date'] is not None):
            query += f"\tWHERE rows.LAST_UPDATE_DATE BETWEEN {kwargs['from_date']} and {kwargs['to_date']};"
        return query
    if query_type == 'valid':
        return f'''
    EXEC sp_describe_first_result_set N'
	SELECT
		*
	FROM
		OPENROWSET(
		BULK ''{bucket}/{table_name}/erpHost={erpHost}/**'',
	    data_source = ''{kwargs['data_source']}'',
	        FORMAT=''{kwargs['format']}''
		) AS nyc';
        '''


def exec_query(conn, *, query_type, bucket, **kwargs):
    query = make_query(query_type, bucket=bucket, **kwargs)
    try:
        cursor = conn.cursor()
        logging.info('Querying:'
                     f'{query}')
        cursor.execute(query)
        for row in cursor:
            if query_type == 'valid':
                info = {
                    'name': row[2],
                    'is_nullable': row[3],
                    'system_type_name': row[5],
                    'max_length': row[6]
                }
            else:
                info = row[0]
            return info
    except Exception as excp:
        logging.error(f'{excp}')
        print(excp)


def get_info(conn, command, **kwargs):
    logging.info(f"{command} with arguments:"
                 f"-t {kwargs['table_name']} -e {kwargs['erpHost']} -db {kwargs['database']}"
                 f"-b {kwargs['bucket_zone']} -s {kwargs['data_source']} -f {format}"
                 # f" -p {kwargs['datapath']} -o {kwargs['dest_file']}"
                 )
    result = {
        'count_raw-zone': None,
        'schema_raw-zone': None,
        'count_clean-zone': None,
        'schema_clean-zone': None
    }
    for zone in kwargs['bucket_zone']:
        result['count_'+zone] = exec_query(conn, query_type='count', bucket=zone, **kwargs)
    for zone in kwargs['bucket_zone']:
        result['schema_'+zone] = exec_query(conn, query_type='valid', bucket=zone, **kwargs)
    return result


if __name__ == '__main__':
    def validate_iso8601(date: str):
        regex = r'^(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9]):([0-5][' \
                r'0-9]):([0-5][0-9])(\.[0-9]+)?(Z|[+-](?:2[0-3]|[01][0-9]):[0-5][0-9])?$ '
        match_iso8601 = re.compile(regex).match
        assert match_iso8601(date) is not None, 'Invalid date format, view --help'
        return date


    # Argument parsing basic Config
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(prog='Table counter',
                                     description='Counts the rows in a table from the extraction process')

    # Argument Parser: count_adls
    subparser = parser.add_subparsers(dest='command', help="""The query will look like this
        SELECT count(*)
        FROM openrowset(
            bulk '{bucket_zone}/{table_name}/erpHost={erpHost}/erpDatabase={database}/**',
            data_source = '{data_source}',
            format = '{format}'
            ) AS rows
        """)
    count_adls_opt = subparser.add_parser('get_info')  # name to use when executed: python program.py <<NAME>>

    # Define the arguments for count_adls
    count_adls_opt.add_argument('-t', '--table_name', type=str, required=True,
                                help='Table to query')
    count_adls_opt.add_argument('-e', '--erpHost', type=str, required=True,
                                help='The ERP host environment')
    count_adls_opt.add_argument('-db', '--database', type=str, required=True,
                                help='ERP database')
    count_adls_opt.add_argument('-b', '--bucket_zone', type=str, required=False, nargs='*',
                                choices=['raw-zone', 'clean-zone'], default=['raw-zone'],
                                help='Bucket zone')
    count_adls_opt.add_argument('-s', '--data_source', type=str, required=True,
                                help='Data source from where to query the data')
    count_adls_opt.add_argument('-p', '--datapath', type=pathlib.Path,
                                required=False, default="/mnt/c/Users/Arkon/wdir/repos/xpl2",
                                help='This is where the directory where the output will be stored')
    count_adls_opt.add_argument('-o', '--dest_file', type=argparse.FileType('w+', encoding='latin-1'),
                                required=False, default='out.txt',
                                help='Name of the output file.')
    count_adls_opt.add_argument('-f', '--format', type=str, required=False,
                                default='PARQUET', choices=['parquet', 'PARQUET'],
                                help='For now the only available format is PARQUET')
    count_adls_opt.add_argument('--from_date', type=str,
                                required=False, default=None,
                                help="Initial date for the query. Must comply with ISO8601")
    count_adls_opt.add_argument('--to_date', type=str,
                                required=False, default=None,
                                help="End date for the query. Must comply with ISO8601")
    args = parser.parse_args()

    # Validate the date arguments
    if not any([args.from_date, args.to_date]):
        pass
    else:
        try:
            assert args.from_date < args.to_date, \
                'The initial date should be older than the end date.'
        except TypeError:
            print('READ THIS → If you are delimiting the query by dates you must use both, --from_date and --to_date. '
                  'Otherwise omit the argument', '\n\n')
            raise

    # Retrieve system variables.
    SERVER = os.environ['SERVER']
    DATABASE = os.environ['DATABASE']
    UID = os.environ['USER']
    PWD = os.environ['PWD']

    # Create the connection
    connection = create_connection(SERVER, DATABASE, UID, PWD)

    # Get the counts
    output_info = None
    kwargs_dict = dict(args._get_kwargs())
    if args.command == 'get_info':
        output_info = get_info(connection, **kwargs_dict)
    elif args.command == 'validate-az-schema':
        pass

    # Print output
    print('———')
    print(f"{args.command}: {output_info}")
    args.dest_file.write(str(output_info))  # Writes out the file
    print(f'Output written to: {args.datapath}/{args.dest_file.name}')
