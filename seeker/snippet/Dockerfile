#date: 2023-09-06T16:58:47Z
#url: https://api.github.com/gists/13f0d36b3a715256442b9ea8cd2b003e
#owner: https://api.github.com/users/vorant94

...
COPY package*.json ./
RUN --mount= "**********"=secret,id=npmrc,target=/usr/local/app/.npmrc \
    npm ci
...