#date: 2024-07-02T16:53:30Z
#url: https://api.github.com/gists/7553f9853e59e841db9e64e645daa97f
#owner: https://api.github.com/users/isaacrlevin

FROM mcr.microsoft.com/dotnet/sdk:8.0-jammy AS build-env
WORKDIR /App

# Copy everything
COPY . ./
# Restore as distinct layers
RUN dotnet restore
# Build and publish a release
RUN dotnet publish -c Release -o out

# Build runtime image
FROM mcr.microsoft.com/dotnet/aspnet:8.0-jammy

WORKDIR /
COPY ../install-pwsh.sh /
RUN chmod -R +x /install-pwsh.sh
RUN /install-pwsh.sh

WORKDIR /App
COPY --from=build-env /App/out .
RUN pwsh ./playwright.ps1 install-deps
RUN pwsh ./playwright.ps1 install
# Change this to your project dll
ENTRYPOINT ["dotnet", "BlazorProject.dll"]