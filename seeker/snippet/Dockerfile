#date: 2022-07-22T17:11:24Z
#url: https://api.github.com/gists/0fe9c7c7e0a35d81fa71dd0a46774eea
#owner: https://api.github.com/users/AlexGalhardo

FROM node:18-alpine
EXPOSE 3000
RUN mkdir /app
WORKDIR /app

COPY package.json index.mjs /app/

ENTRYPOINT ["node", "index.mjs"]