#date: 2021-12-08T17:16:13Z
#url: https://api.github.com/gists/73ed3944f0b682ea80fa8a9c85aaa9cf
#owner: https://api.github.com/users/camilomelges

FROM node:lts-alpine as build-stage
WORKDIR /app
COPY package*.json ./
RUN CYPRESS_INSTALL_BINARY=0 npm install
COPY . .
RUN npm run build -- --mode production

FROM nginx:stable-alpine as production-stage
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build-stage /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]