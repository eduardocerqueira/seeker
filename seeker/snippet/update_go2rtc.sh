#date: 2025-06-02T16:42:51Z
#url: https://api.github.com/gists/099be443eb6a5fc77dac68dd1d7277c4
#owner: https://api.github.com/users/felipecrs

#!/bin/bash

set -euxo pipefail

version="${1:-latest}"

case $(uname -m) in
    x86_64) arch="amd64" ;;
    aarch64) arch="arm64" ;;
    *)
        echo "unsupported arch" >&2
        exit 1
        ;;
esac

cd /tmp
if [[ "${version}" == master ]]; then
    curl -fL -# --output go2rtc.zip \
        "https://nightly.link/AlexxIT/go2rtc/workflows/build/master/go2rtc_linux_${arch}.zip"
    unzip -o go2rtc.zip
    rm -f go2rtc.zip
elif [[ "${version}" == seydx ]]; then
    curl -fL -# --output go2rtc.zip \
        "https://nightly.link/seydx/go2rtc/workflows/build/dev/go2rtc_linux_${arch}.zip"
    unzip -o go2rtc.zip
    rm -f go2rtc.zip
elif [[ "${version}" == latest ]]; then
    curl -fL -# --output go2rtc \
        "https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_linux_${arch}"
    chmod +x go2rtc
else
    curl -fL -# --output go2rtc \
        "https://github.com/AlexxIT/go2rtc/releases/download/v${version##v}/go2rtc_linux_${arch}"
    chmod +x go2rtc
fi

./go2rtc --version

# For WebRTC integration
if (shopt -s nullglob; echo /config/go2rtc-*.*.*) | grep -q .; then
    cp -f go2rtc /config/go2rtc
    ha core restart
fi

# Frigate 0.16 Beta onwards
if [[ -d /addon_configs/ccab4aaf_frigate-beta ]]; then
    cp -f go2rtc /addon_configs/ccab4aaf_frigate-beta/go2rtc
    ha addons restart ccab4aaf_frigate-beta
fi

# For when Frigate 0.16 Beta becomes stable
if [[ -d /addon_configs/ccab4aaf_frigate ]]; then
    cp -f go2rtc /addon_configs/ccab4aaf_frigate/go2rtc
    ha addons restart ccab4aaf_frigate
fi

# For Frigate 0.15 downwards
if [[ -f /config/frigate.yml || -f /config/frigate.yaml ]]; then
    cp -f go2rtc /config/go2rtc
    ha addons restart ccab4aaf_frigate
fi

rm -f go2rtc
