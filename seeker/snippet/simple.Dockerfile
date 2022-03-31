#date: 2022-03-31T17:14:50Z
#url: https://api.github.com/gists/04b2cad14b00e5ffe8ec96a3afbb34fb
#owner: https://api.github.com/users/AradAlvand

FROM node:lts-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . ./

RUN npm run build

EXPOSE 3000

CMD [ "node", "build" ]
