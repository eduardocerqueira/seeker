#date: 2021-11-02T17:17:55Z
#url: https://api.github.com/gists/bb0ae65f3d16998489bb7c917d949cc7
#owner: https://api.github.com/users/jjeanjacques10

echo " | SERVER CONFIG SCRIPT |"
echo "Install GIT"
sudo apt-get install git
sudo apt-get install build-essential
sudo apt-get install curl openssl libssl-dev
echo "======= Install NVM ======="
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.37.2/install.sh | bash
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
echo "======= Install NodeJS version 14.16.0 ======="
nvm install v14.16.0
nvm alias default v14.16.0
nvm use v14.16.0
echo "======= Install Yarn ======="
npm install --global yarn
yarn --version
echo "======= Install Docker ======="
sudo apt-get update
sudo apt install docker.io
echo "======= Docker version ======="
sudo docker --version
echo "======= Install Docker Compose ======="
sudo curl -L "https://github.com/docker/compose/releases/download/1.28.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
echo "======= Docker Compose version ======="
docker-compose --version
echo "======= Install Typescript ======="
npm install --global typescript
echo "======= Install pm2 ======="
npm install --global pm2
pm2 install typescript
pm2 install @types/node
echo "======= FINISHED ======="