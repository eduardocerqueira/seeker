#date: 2021-09-27T17:13:07Z
#url: https://api.github.com/gists/8f8f888fd170351c3d9f7b1c3b001cca
#owner: https://api.github.com/users/srimaln91

apt  install docker.io
docker volume create gitlab-runner-config
docker run -d --name gitlab-runner --restart always     -v /var/run/docker.sock:/var/run/docker.sock     -v gitlab-runner-config:/etc/gitlab-runner     gitlab/gitlab-runner:latest
docker run --rm -it -v gitlab-runner-config:/etc/gitlab-runner gitlab/gitlab-runner:latest register
