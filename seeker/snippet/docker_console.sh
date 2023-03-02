#date: 2023-03-02T17:09:03Z
#url: https://api.github.com/gists/7178b5065bd6f684aef5fba210e95b18
#owner: https://api.github.com/users/alioualarbi

# Don't forget to change <text> with the search term
#!/bin/bash
DOCKER_CONTAINER_NAME="container name or other text"
DOCKER_CONTAINER="$(docker ps | grep $DOCKER_CONTAINER_NAME | awk '{print $1;}')"
DOCKER_RUN_CONSOLE="docker exec -it $DOCKER_CONTAINER /bin/bash"

ascii_name() {
cat <<"EOT"

     _               _                                                  _       
    | |             | |                                                | |      
  __| |  ___    ___ | | __  ___  _ __    ___   ___   _ __   ___   ___  | |  ___ 
 / _` | / _ \  / __|| |/ / / _ \| '__|  / __| / _ \ | '_ \ / __| / _ \ | | / _ \
| (_| || (_) || (__ |   < |  __/| |    | (__ | (_) || | | |\__ \| (_) || ||  __/
 \__,_| \___/  \___||_|\_\ \___||_|     \___| \___/ |_| |_||___/ \___/ |_| \___|
                                                                                
                                                                                                                                                           
EOT
}
ascii_name
eval $DOCKER_RUN_CONSOLE