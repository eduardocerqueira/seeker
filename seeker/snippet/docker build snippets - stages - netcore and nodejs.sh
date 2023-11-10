#date: 2023-11-10T17:06:59Z
#url: https://api.github.com/gists/9ace2f7ce02bfd1605c3a0e920553413
#owner: https://api.github.com/users/jrichardsz

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS net_build
WORKDIR /app
# etc
# generate the dll here

FROM node:18 as spa_build
WORKDIR /app/src
RUN npm install
RUN npm run build
# etc
# generate the spa static build here

FROM mcr.microsoft.com/dotnet/aspnet:7.0
WORKDIR /net

# get the dll and other files from build stage
COPY --from=net_build /tmp/publish /net

# get the react/angular/vue build files from spa stage
COPY --from=spa_build /tmp/build /net/spa

# etc
# run your dll with dotnet here

