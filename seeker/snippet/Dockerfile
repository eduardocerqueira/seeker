#date: 2023-05-30T16:46:48Z
#url: https://api.github.com/gists/bb29700d8ab4073e0efa4a5793639bee
#owner: https://api.github.com/users/jancel

FROM debian:10

################################################################
# Install bulk copy program (bcp)
################################################################
RUN apt install -y gnupg2 apt-transport-https wget curl
RUN wget -q -O- https://packages.microsoft.com/keys/microsoft.asc | \
  gpg --dearmor | tee /usr/share/keyrings/microsoft.gpg > /dev/null 2>&1
RUN echo "deb [signed-by=/usr/share/keyrings/microsoft.gpg arch=amd64,armhf,arm64] https://packages.microsoft.com/debian/10/prod buster main" | \
  tee /etc/apt/sources.list.d/prod.list
RUN apt-get update && env ACCEPT_EULA=Y apt-get install -y mssql-tools unixodbc-dev
################################################################