#date: 2026-02-09T17:32:52Z
#url: https://api.github.com/gists/2d8044a49d28aababd617137c7efbe9c
#owner: https://api.github.com/users/MonstraG

#!/bin/zsh

# Make a temporary working directory
echo "Creating temporary working directory..."
rm -rf /tmp/proton-ge-custom
mkdir /tmp/proton-ge-custom
cd /tmp/proton-ge-custom

# fetch releases info
proton_releases="$(curl -s https://api.github.com/repos/GloriousEggroll/proton-ge-custom/releases/latest)"

# Download the checksum for the latest release
echo "Fetching checksum URL..."
checksum_url=$(echo $proton_releases | grep browser_download_url | cut -d\" -f4 | grep .sha512sum)
checksum_name=$(basename $checksum_url)
tarball_url=$(echo $proton_releases | grep browser_download_url | cut -d\" -f4 | grep .tar.gz)
tarball_name=$(basename $tarball_url)
echo "Downloading checksum: $checksum_name..."
curl -# -L $checksum_url -o $checksum_name

# get just the archive name, without the extension
# this will be the folder where the arhive is extracted to
proton_name="${tarball_name%.tar.gz}"

old_checksum="$HOME/.steam/steam/compatibilitytools.d/$proton_name/$checksum_name"

# compare saved checksum file to see if we need updating
if (cmp $checksum_name $old_checksum) {
    echo "Already on the latest: $proton_name"
    exit 0
}

# Download the tarball for the latest release
echo "Downloading tarball: $tarball_name..."
curl -# -L $tarball_url -o $tarball_name

# Verify the downloaded tarball with the downloaded checksum
echo "Verifying tarball $tarball_name with checksum $checksum_name..."
sha512sum -c $checksum_name
# If result the verification succeeds, continue

# Make a Steam compatibility tools folder if it does not exist
echo "Creating a Steam compatibility tools folder if it does not exist..."
mkdir -p ~/.steam/steam/compatibilitytools.d


# Extract the GE-Proton tarball to the Steam compatibility tools folder
echo "Extracting $tarball_name to the Steam compatibility tools folder..."
tar -xf $tarball_name -C ~/.steam/steam/compatibilitytools.d/

mv $checksum_name ~/.steam/steam/compatibilitytools.d/$proton_name/$checksum_name

echo "Done"
