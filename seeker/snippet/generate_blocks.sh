#date: 2023-07-03T16:38:31Z
#url: https://api.github.com/gists/1695c8b5eda3f6267be9418c0b53053f
#owner: https://api.github.com/users/fogmoon

# Script to generate a new block every minute
# Put this script at the root of your unpacked folder
#!/bin/bash

echo "Generating a block every minute. Press [CTRL+C] to stop.."

address=`./bin/bitcoin-cli getnewaddress`

while :
do
        echo "Generate a new block `date '+%d/%m/%Y %H:%M:%S'`"
        ./bin/bitcoin-cli generatetoaddress 1 $address
        sleep 60
done
