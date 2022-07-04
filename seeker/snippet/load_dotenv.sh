#date: 2022-07-04T03:30:37Z
#url: https://api.github.com/gists/6a32f8cd70f1de5b37a409d4205f8b1c
#owner: https://api.github.com/users/hoangnvhybrid

if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi