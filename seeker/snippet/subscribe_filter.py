#date: 2021-11-29T17:02:07Z
#url: https://api.github.com/gists/85b5153e0c18b1e086d08d6fc894ca89
#owner: https://api.github.com/users/jac18281828

from blocknative.stream import Stream as BNStream
import json,sys,traceback,logging,os

monitor_address = '0x7a250d5630b4cf539739df2c5dacb4c659f2488d'

async def txn_handler(txn, unsubscribe):
    print(json.dumps(txn, indent=4))

def api_key_or_env():
    api_key = os.getenv('BN_API_KEY', None)
    if api_key is None or len(api_key) == 0:
        if len(sys.argv) == 1:
            print('%s apikey' % sys.argv[0])
            sys.exit(1)
        else:
            apikeyfile = sys.argv[1]            
            logging.info('loading apikey from %s' % apikeyfile)
            with open(apikeyfile, 'r') as apikey:
                api_key = apikey.readline().rstrip().lstrip()
                return api_key
    else:
        logging.info('found BN_API_KEY env var')
        return api_key
        

if __name__ == '__main__':
    try:
        logging.basicConfig(level=logging.INFO)

        api_key = api_key_or_env()

        filter = {
            'internalTransactions.input': '0x',
            '_start': True
        }
        
        stream = BNStream(api_key)
        stream.subscribe_address(monitor_address, txn_handler, filters=[ filter ])

        stream.connect()
    except Exception as e:
        print('API Failed: %s' % str(e))
        traceback.print_exc(e)
