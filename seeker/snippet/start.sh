#date: 2022-09-28T17:17:15Z
#url: https://api.github.com/gists/8b2dc08f091f14122c9b47f53968ad94
#owner: https://api.github.com/users/ps-jessejjohnson

#!/usr/local/bin/bash

if [ ! -d logs ]; then
    mkdir logs
fi

cd ./rosetta

echo sendToScout/rosetta
PROCESS=sendToScout rosetta > ../logs/rosetta-in.logs 2>&1 &

cd ../scout

echo exchange:director
nohup rake exchange:director > ../logs/director.log 2>&1 &
sleep 1

echo proxy:interceptor
nohup rake proxy:interceptor > ../logs/proxy.log 2>&1 &
sleep 1

echo micro:product:real_time:consumer
nohup rake micro:product:real_time:consumer > ../logs/product.log 2>&1 &
sleep 1

echo micro:marketplace:real_time:consumer
nohup rake micro:marketplace:real_time:consumer > ../logs/marketplace.log  2>&1 &
sleep 1

echo exchange:post_director
nohup rake exchange:post_director > ../logs/post.log 2>&1 &
sleep 1

cd ../premo

echo premo curl
PREMO_ENGINE=curl PORT=3000 yarn start > ../logs/curl.log 2>&1 &
sleep 1

echo premo puppeteer
PREMO_ENGINE=puppeteer PORT=3001 yarn start > ../logs/puppeteer.log 2>&1 &
sleep 1

echo premo nightmare
PREMO_ENGINE=nightmare PORT=3002 yarn start > ../logs/nightmare.log 2>&1 &
sleep 1

cd ../rosetta
echo sentToPS/rosetta
PROCESS=sendToPS rosetta > ../logs/rosetta-out.logs 2>&1 &
cd ..

echo ===== Pipeline active =====