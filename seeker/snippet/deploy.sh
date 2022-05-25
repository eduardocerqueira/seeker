#date: 2022-05-25T17:17:44Z
#url: https://api.github.com/gists/e5be621342ce85f6ffe24e6a839d60ff
#owner: https://api.github.com/users/ganiirsyadi

#!/bin/sh

ssh -o StrictHostKeyChecking=no ubuntu@staging.ajaib.me << 'ENDSSH'
  cd app
  aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 781315367039.dkr.ecr.ap-southeast-1.amazonaws.com
  docker pull 781315367039.dkr.ecr.ap-southeast-1.amazonaws.com/ajaibdex-backend:staging
  docker compose up -d
  docker exec -e MIKRO_ORM_CLI_USE_TS_NODE=false backend node node_modules/.bin/mikro-orm migration:up
ENDSSH
