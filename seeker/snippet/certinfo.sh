#date: 2025-11-05T17:12:30Z
#url: https://api.github.com/gists/750bfdf26907d2007f7ed5fe83b07e4f
#owner: https://api.github.com/users/andrewhaller

#!/bin/sh

### pre-requisites: jq, wget, openssl, egrep, sed, awk, tr

OPENID_CONFIG_URL='https: "**********"
SECURE_HOST=$(wget -qO- $OPENID_CONFIG_URL | jq -r '.jwks_uri | split("/")[2]'); \
SECURE_PORT=443; \
X509FINGER='-fingerprint -sha1'; \
X509ISSUER='-subject -issuer'; \
X509COMMAND="$X509FINGER"; \
FINGER_PREFIX=''; \
ISSUER_PREFIX='INFO | '; \
while read line; do \
  if [ "${line//END}" != "$line" ]; then \
    txt="$txt$line\n"; \
    printf -- "$txt" | openssl x509 ${X509COMMAND:--fingerprint -sha1} -noout; \
    txt=""; \
  else \
    txt="$txt$line\n"; \
  fi; \
done \
< <(openssl s_client \
    -servername $SECURE_HOST \
    -showcerts \
    -connect $SECURE_HOST:${SECURE_PORT:-443} \
  < /dev/null 2>/dev/null \
  | awk '/BEGIN/,/END/{ if(/BEGIN/){a++}; print}') \
| { \
  echo $X509COMMAND | egrep -vq '^-(subject|issuer)' \
    && { \
        awk -F'=' '{print $2}' \
        | sed 's/://g' \
        | tr '[:upper:]' '[:lower:]' \
        | xargs -I {} printf "\033[1m%s\033[0m%s\n" "$FINGER_PREFIX" "{}"; \
       } \
    || xargs -I {} printf "\033[1m%s\033[0m%s\n" "$ISSUER_PREFIX" "{}"; \
  }I {} printf "\033[1m%s\033[0m%s\n" "$ISSUER_PREFIX" "{}"; \
  }