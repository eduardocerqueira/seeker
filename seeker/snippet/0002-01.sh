#date: 2025-01-14T17:03:07Z
#url: https://api.github.com/gists/4afc9350ffa282c2b81f396d21bcb9fb
#owner: https://api.github.com/users/sercangezer-linuxdevopscomtr

# AMD64 platform
curl -sSfL -o longhornctl https://github.com/longhorn/cli/releases/download/v1.7.2/longhornctl-linux-amd64
chmod +x longhornctl

# Eksiklikleri kontrol edelim
./longhornctl check preflight

# Eksik olan varsa kuralım
./longhornctl install preflight