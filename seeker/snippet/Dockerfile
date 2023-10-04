#date: 2023-10-04T17:07:33Z
#url: https://api.github.com/gists/ccf1839031979f7464b1895cfa04758c
#owner: https://api.github.com/users/jeff-silva

FROM node:18.16-alpine
WORKDIR /app

COPY . .

EXPOSE $NUXT_PORT

CMD yarn install && yarn dev