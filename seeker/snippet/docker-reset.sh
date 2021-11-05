#date: 2021-11-05T16:52:23Z
#url: https://api.github.com/gists/967a1730327a2474cad14c587172570b
#owner: https://api.github.com/users/imcharliesparks

docker-sync stop
docker-sync clean
docker-sync start

// then, in terminal inside Website
docker cp ~/.ssh app:/home/www/
docker exec -u 0 -it app bash
chown -R www:www /home/www/.ssh