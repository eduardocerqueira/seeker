#date: 2021-12-14T16:58:30Z
#url: https://api.github.com/gists/b24b9c09b770e1b353df19b82b364286
#owner: https://api.github.com/users/karanokuri

FROM debian:bullseye-slim AS build-env

ENV DEBCONF_NOWARNINGS=yes

RUN apt-get update \
 && apt-get install --no-install-suggests --no-install-recommends --yes \
      privoxy \
 && apt-get -y clean

FROM gcr.io/distroless/base-debian11
COPY --from=build-env \
  /usr/lib/x86_64-linux-gnu/libmbedcrypto.so.* \
  /usr/lib/x86_64-linux-gnu/libmbedx509.so.* \
  /usr/lib/x86_64-linux-gnu/libmbedtls.so.* \
  /usr/lib/x86_64-linux-gnu/libpcreposix.so.* \
  /usr/lib/x86_64-linux-gnu/libbrotlicommon.so.* \
  /usr/lib/x86_64-linux-gnu/libbrotlidec.so.* \
  /usr/lib/x86_64-linux-gnu/libbrotlienc.so.* \
  /usr/lib/x86_64-linux-gnu/
COPY --from=build-env \
  /lib/x86_64-linux-gnu/libpcre.so.* \
  /lib/x86_64-linux-gnu/libz.so.* \
  /lib/x86_64-linux-gnu/
COPY --from=build-env /usr/sbin/privoxy /usr/sbin/
ENTRYPOINT ["/usr/sbin/privoxy"]
