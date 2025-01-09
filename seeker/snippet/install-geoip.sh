#date: 2025-01-09T17:11:30Z
#url: https://api.github.com/gists/97ba2079a3da4b819bb01507d6793084
#owner: https://api.github.com/users/Rush-er

#!/bin/bash

# Exit on error
set -e

# Install prerequisites
echo -e "\033[32mInstalling prerequisites...\033[0m"
apt update
apt -y install curl unzip perl libtext-csv-xs-perl libmoosex-types-netaddr-ip-perl pkg-config

# Install Linux headers
echo ""
echo -e "\033[32mInstalling current Linux headers...\033[0m"
apt install linux-headers-$(uname -r) -y

# Download and install xtables-addons
XTA_VERSION="3.25"
XTA_URL="https://inai.de/files/xtables-addons/xtables-addons-${XTA_VERSION}.tar.xz"
XTA_DIR="/tmp/xtables-addons-${XTA_VERSION}"

echo ""
echo -e "\033[32mDownloading xtables-addons version ${XTA_VERSION}...\033[0m"
curl -L ${XTA_URL} -o /tmp/xtables-addons-${XTA_VERSION}.tar.xz

echo ""
echo -e "\033[32mExtracting xtables-addons...\033[0m"
tar xf /tmp/xtables-addons-${XTA_VERSION}.tar.xz -C /tmp/

echo ""
echo -e "\033[32mBuilding and installing xtables-addons...\033[0m"
cd ${XTA_DIR}
./configure
make
make install
cd ..
rm -rf ${XTA_DIR} /tmp/xtables-addons-${XTA_VERSION}.tar.xz

# GeoIP database update
echo ""
echo -e "\033[32mPreparing to update GeoIP database...\033[0m"
GEOIP_DIR="/usr/share/xt_geoip/"
DATE=$(date +'%Y-%m')
GEOIP_URL="https://download.db-ip.com/free/dbip-country-lite-${DATE}.csv.gz"
GEOIP_CSV_GZ_FILE="${GEOIP_DIR}dbip-country-lite-${DATE}.csv.gz"
GEOIP_CSV_FILE="${GEOIP_DIR}dbip-country-lite-${DATE}.csv"

# Create the GeoIP directory if it doesn't exist
mkdir -p ${GEOIP_DIR}

echo ""
echo -e "\033[34mPlease download the GeoIP database using your web browser due to Cloudflare restrictions.\033[0m"
echo -e "\033[34mDownload URL:\033[0m ${GEOIP_URL}"
echo -e "\033[34mThen upload the file to this path:\033[0m ${GEOIP_CSV_GZ_FILE}"

# Wait until the file is present
while true; do
    echo "Press enter to continue after uploading the file..."
    read -r _  # Waits for user to press enter, ignoring the input
    if [ -f "${GEOIP_CSV_GZ_FILE}" ]; then
        echo "File detected. Continuing with the process."
        break # Exit the loop if file is found
    else
        echo -e "\033[33mFile not found. Please ensure the file is saved to ${GEOIP_DIR} and try again.\033[0m"
    fi
done

echo ""
echo -e "\033[32mExtracting GeoIP CSV file...\033[0m"
cd ${GEOIP_DIR}
gunzip "${GEOIP_CSV_GZ_FILE}" -f

echo ""
echo -e "\033[32mLocating and running xt_geoip_build...\033[0m"

# Define possible locations for xt_geoip_build
POSSIBLE_LOCATIONS=(
    "/usr/lib/xtables-addons/xt_geoip_build"
    "/usr/libexec/xtables-addons/xt_geoip_build"
    "/usr/local/lib/xtables-addons/xt_geoip_build"
    "/usr/local/libexec/xtables-addons/xt_geoip_build"
)

GEOIP_BUILD=""
for location in "${POSSIBLE_LOCATIONS[@]}"; do
    if [ -f "$location" ]; then
        GEOIP_BUILD="$location"
        break
    fi
done

if [ -z "$GEOIP_BUILD" ]; then
    echo -e "\033[31mError: Could not find xt_geoip_build script in any known location\033[0m"
    echo "Searching for xt_geoip_build in the system..."
    FOUND_PATH=$(find / -name "xt_geoip_build" 2>/dev/null)
    
    if [ -n "$FOUND_PATH" ]; then
        echo -e "\033[32mFound xt_geoip_build at: $FOUND_PATH\033[0m"
        GEOIP_BUILD="$FOUND_PATH"
    else
        echo -e "\033[31mFatal: xt_geoip_build script not found anywhere in the system\033[0m"
        exit 1
    fi
fi

echo -e "\033[32mBuilding the GeoIP database with xtables-addons...\033[0m"
mv ${GEOIP_CSV_FILE} dbip-country-lite.csv
"$GEOIP_BUILD" -D /usr/share/xt_geoip *.csv
rm -f ${GEOIP_CSV_FILE}

# Dependency fix and module load
echo ""
echo -e "\033[32mFixing dependencies with depmod and loading xt_geoip module...\033[0m"
depmod
modprobe xt_geoip

# Verify module loading
if lsmod | grep -q "^xt_geoip"; then
    echo -e "\033[32mxt_geoip module successfully loaded.\033[0m"
else
    echo -e "\033[31mFailed to load xt_geoip module. Please check manually.\033[0m"
fi

echo ""
echo "----------------------------------------"
echo ""
echo -e "\033[32mComplete!\033[0m"
echo ""
echo "iptables is now ready to use with GeoIP module!"
echo ""

# Tabbed iptables lines
echo -e "\033[34mTo block a specific country, use the following command:\033[0m"
echo -e "\tiptables -I INPUT -m geoip --src-cc XX -j DROP"
echo -e "\033[33mReplace XX with the country code you want to block.\033[0m"
echo ""
echo -e "\033[34mTo block all countries except your own, use the following command:\033[0m"
echo -e "\tiptables -I INPUT -m geoip ! --src-cc YY -j DROP"
echo -e "\033[33mReplace YY with your country code.\033[0m"
echo ""