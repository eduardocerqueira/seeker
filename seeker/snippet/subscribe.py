#date: 2021-11-02T17:00:18Z
#url: https://api.github.com/gists/7e7d9fb5f5ed75b9e14b570b4e5fe687
#owner: https://api.github.com/users/jac18281828

from blocknative.stream import Stream as BNStream
import json,sys

address = '0x7a250d5630b4cf539739df2c5dacb4c659f2488d'

async def txn_handler(txn, unsubscribe):
    print(json.dumps(txn, indent=4))

if __name__ == '__main__':
    try:
        if len(sys.argv) == 1:
            print('{} apikey' % sys.argv[0])
        else:
            apikeyfile = sys.argv[1]
            with open(apikeyfile, 'r') as apikey:
                keystring = apikey.readline().rstrip().lstrip()
                print(keystring)
                stream = BNStream(keystring)
                stream.subscribe_address(address, txn_handler)
                stream.connect()
    except Exception as e:
        print('Failed: ' + str(e))
