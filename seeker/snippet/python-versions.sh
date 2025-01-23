#date: 2025-01-23T17:06:32Z
#url: https://api.github.com/gists/aabe80601a69287766cb8a4307e0910e
#owner: https://api.github.com/users/turicas

#!/bin/bash

cat <<'EOF'> check-python-version.sh
#!/bin/bash

apt update
apt install -y python3
source /etc/os-release
echo "${NAME} | ${VERSION_ID} | ${VERSION_CODENAME} | $(python3 --version)"
EOF

# Debian
# I had problems executing this in stretch and jessie, so skipped
for version in bookworm bullseye buster; do
  image="debian:${version}"
  echo "-----> $image"
  docker run \
  	  -v $(pwd)/check-python-version.sh:/check-python-version.sh \
  	  --rm \
  	  "$image" \
  	  bash -c 'chmod +x /check-python-version.sh && /check-python-version.sh'
done

# By the time I made this script, only the LTS versions were available in APT
for version in 25.04 24.10 24.04 23.10 22.04 20.04 18.04; do
  image="ubuntu:${version}"
  echo "-----> $image"
  docker run \
  	  -v $(pwd)/check-python-version.sh:/check-python-version.sh \
  	  --rm \
  	  "$image" \
  	  bash -c 'chmod +x /check-python-version.sh && /check-python-version.sh'
done
