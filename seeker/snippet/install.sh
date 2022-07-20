#date: 2022-07-20T17:15:19Z
#url: https://api.github.com/gists/9c1758936118a7f677ccf03f44ae3927
#owner: https://api.github.com/users/tiagofrancafernandes

curl -fsSL https://get.docker.com -o /tmp/get-docker.sh && sudo sh /tmp/get-docker.sh && \
    sudo usermod -aG docker $USER && newgrp docker && \
    newgrp docker && \
    sudo curl -L "github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/bin/docker-compose && \
    sudo chmod +x /usr/bin/docker-compose