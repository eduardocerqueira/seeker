#date: 2021-10-22T17:15:27Z
#url: https://api.github.com/gists/fc0557c6c095c75c6358f39130f4b3a1
#owner: https://api.github.com/users/vinayski

docker build -t prenode .                                                                                                                                 <aws:default>
[+] Building 1.8s (10/10) FINISHED
 => [internal] load build definition from Dockerfile                                                                                                                                                                                   0.0s
 => => transferring dockerfile: 383B                                                                                                                                                                                                   0.0s
 => [internal] load .dockerignore                                                                                                                                                                                                      0.0s
 => => transferring context: 2B                                                                                                                                                                                                        0.0s
 => [internal] load metadata for docker.io/presearch/node:latest                                                                                                                                                                       1.7s
 => [auth] presearch/node:pull token for registry-1.docker.io                                                                                                                                                                          0.0s
 => [1/4] FROM docker.io/presearch/node:latest@sha256:6d5ff4801068d630922d6384d531586a1c75937d9cc257441bf1d8b55fe778da                                                                                                                 0.0s
 => => resolve docker.io/presearch/node:latest@sha256:6d5ff4801068d630922d6384d531586a1c75937d9cc257441bf1d8b55fe778da                                                                                                                 0.0s
 => [internal] load build context                                                                                                                                                                                                      0.0s
 => => transferring context: 2.23kB                                                                                                                                                                                                    0.0s
 => CACHED [2/4] RUN mkdir -p /app/node/.keys                                                                                                                                                                                          0.0s
 => CACHED [3/4] COPY ./id_rsa      /app/node/.keys/id_rsa                                                                                                                                                                             0.0s
 => CACHED [4/4] COPY ./id_rsa.pub  /app/node/.keys/id_rsa.pub                                                                                                                                                                         0.0s
 => exporting to image                                                                                                                                                                                                                 0.0s
 => => exporting layers                                                                                                                                                                                                                0.0s
 => => writing image sha256:8ca9d5afa4278cfece08eb754518f78c1ffe5ca8ef413c6787bdb8e98f93271b                                                                                                                                           0.0s
 => => naming to docker.io/library/prenode
 
 
docker images                                                                                                                                              <aws:default>
REPOSITORY   TAG       IMAGE ID       CREATED      SIZE
prenode      latest    8ca9d5afa427   3 days ago   67.7MB