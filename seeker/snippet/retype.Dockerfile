#date: 2021-12-17T16:59:28Z
#url: https://api.github.com/gists/22c40a2e3d65e13c81903e3dd6874c42
#owner: https://api.github.com/users/Genzer

FROM node:14.18.2-stretch-slim as build

# This is to suppress the error
# "Couldn't find a valid ICU package installed on the system.".
ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1

# These packages has to be install as `retype` is dynamically linked
# with several libs.
ARG PACKAGES="\
  libgssapi-krb5-2 \
  libssl-dev \
"

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    libgssapi-krb5-2 \
    libssl-dev \
  && rm -rf /var/lib/apt/lists/* \
  && npm install -g retypeapp