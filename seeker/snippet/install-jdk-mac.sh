#date: 2025-12-12T16:48:13Z
#url: https://api.github.com/gists/9bcf841a6702edbef10fd233e04eb37e
#owner: https://api.github.com/users/tastyone

mkdir -p ~/.cache/jdk-17.0.17
curl -L -o temurin.tar.gz https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.17%2B10/OpenJDK17U-jdk_aarch64_mac_hotspot_17.0.17_10.pkg
tar -xzf temurin.tar.gz -C ~/.cache/jdk-17.0.17 --strip-components=1
rm temurin.tar.gz