#date: 2024-05-06T17:11:25Z
#url: https://api.github.com/gists/28154474441720e45e7e296375fc3383
#owner: https://api.github.com/users/burgil

ps aux | grep "MyMainWebsite" | grep -v grep | awk '{print $2}' | xargs -I{} kill -9 {}
cd ..
git fetch
git pull
cd website
nohup npm start &