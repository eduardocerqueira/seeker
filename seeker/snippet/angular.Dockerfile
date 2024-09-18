#date: 2024-09-18T17:05:04Z
#url: https://api.github.com/gists/124d04536cdeca0cc709c6b43ffd9871
#owner: https://api.github.com/users/Riko07br

# Build angular------------------
FROM node:lts-alpine3.20 as build

RUN npm install -g @angular/cli

WORKDIR /app

COPY package.json .

RUN npm install

COPY . .

RUN npm run build

# NGINX runner---------------
FROM nginx:1.21-alpine

COPY --from=build /app/dist/frontend/browser /usr/share/nginx/html

COPY ./nginx.conf  /etc/nginx/conf.d/default.conf

EXPOSE 80 4200

CMD ["nginx", "-g", "daemon off;"]