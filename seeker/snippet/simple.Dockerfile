#date: 2023-10-27T17:01:47Z
#url: https://api.github.com/gists/f763c7696c6575d00ba77cd2e5f423a2
#owner: https://api.github.com/users/DiegoHinA

FROM node:16-alpine

RUN mkdir -p /app

WORKDIR /app

COPY package.json /app

RUN yarn install

COPY . /app

RUN yarn build

EXPOSE 3000

CMD [ "yarn", "start" ]

# Super pesada +1GB