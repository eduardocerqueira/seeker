#date: 2023-01-18T16:44:49Z
#url: https://api.github.com/gists/660293384774c77fe8e98afc4390ef22
#owner: https://api.github.com/users/dsakovych

# build image
docker build --tag namespace/name:tag .

# run docker container with options
docker run --detach \
--name container_name \
--publish 80:8888 \
--runtime=nvidia \
--volume /path/to/local:/path/container \
namespace/name:tag