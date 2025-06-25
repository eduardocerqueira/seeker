#date: 2025-06-25T17:05:47Z
#url: https://api.github.com/gists/25cf5d69100d65703a64bc083ae357c6
#owner: https://api.github.com/users/mikegilchrist

#!/bin/bash
#
# Clonezilla GRUB Boot Entry Installer
# ------------------------------------
# Downloads Clonezilla Live ISO and adds it as a GRUB boot menu entry.
#
# Main Author: Mike Gilchrist <github.com/mikegilchrist>
# Collaborator: Bishop Bash (ChatGPT, OpenAI)
# Original Basis: https://www.q4os.org/forum/viewtopic.php?id=4901
# Date: 2025-06-25
# Version: 1.4.1
#
# NOTE: This script assumes a Debian-based system (e.g., Ubuntu, Q4OS, etc.)
#       that supports `.deb` packages and has dpkg installed.
#

# ==== Root Check ====
if [ "$EUID" -ne 0 ]; then
  echo " !! This script must be run as root."
  exit 1
fi

# ==== Fetch and Present Clonezilla Versions (no duplicates) ====
echo " # Fetching list of Clonezilla stable versions..."
versions=$(curl -s https://sourceforge.net/projects/clonezilla/files/clonezilla_live_stable/ | \
  grep -oP '(?<=href="/projects/clonezilla/files/clonezilla_live_stable/)[0-9]+\.[0-9]+\.[0-9]+-[0-9]+(?=/)' | \
  sort -Vr | uniq | head -n 10)

if [ -z "$versions" ]; then
  echo " !! Failed to fetch version list from SourceForge."
  exit 1
fi

mapfile -t version_list <<< "$versions"
default_version="${version_list[0]}"

echo
echo "Latest Clonezilla Stable Releases (includes both amd64 and i686 builds):"
for i in "${!version_list[@]}"; do
  printf " [%d] %s%s\n" "$((i+1))" "${version_list[$i]}" \
    "$( [[ "${version_list[$i]}" == "$default_version" ]] && echo " (latest)" )"
done

echo
read -p "Select release number [default: 1]: " selection
selection=${selection:-1}

if ! [[ "$selection" =~ ^[0-9]+$ ]] || (( selection < 1 || selection > ${#version_list[@]} )); then
  echo " !! Invalid selection."
  exit 1
fi

CZVER="${version_list[$((selection-1))]}"
echo " -> Selected Clonezilla version: $CZVER"

# ==== System Info ====
osarch=$(dpkg --print-architecture)
uuid=$(lsblk -o MOUNTPOINT,UUID | awk '$1 == "/" {print $2}')
current_locale=$(locale | grep LANG= | cut -d'=' -f2)
keyboard_layout=$(setxkbmap -query | grep layout | awk '{print $2}')

echo
echo " # Detected System Configuration:"
echo "  > Architecture: $osarch"
echo "  > Root UUID:    $uuid"
echo "  > Locale:       $current_locale"
echo "  > Keyboard:     $keyboard_layout"
echo
read -p " Proceed with download and installation? (y/N): " confirm
[[ "$confirm" != "y" && "$confirm" != "Y" ]] && echo " Aborted." && exit 0

# ==== Build ISO Name and URL ====
if [ "$osarch" = "amd64" ]; then
  zillaiso="clonezilla-live-$CZVER-amd64.iso"
else
  zillaiso="clonezilla-live-$CZVER-i686.iso"
fi

BASE_URL="https://sourceforge.net/projects/clonezilla/files/clonezilla_live_stable/$CZVER"
iso_url="$BASE_URL/$zillaiso/download"

# ==== Download Directory Setup ====
mkdir -p /iso
cd /iso || exit 1

# ==== Check for Existing ISO ====
existing_iso=$(ls clonezilla-live-*-amd64.iso 2>/dev/null | head -n 1)

if [[ -n "$existing_iso" ]]; then
  existing_ver=$(echo "$existing_iso" | sed -n 's/clonezilla-live-\(.*\)-amd64\.iso/\1/p')
  echo " # Detected existing Clonezilla ISO: $existing_iso (version: $existing_ver)"

  if [[ "$existing_ver" == "$CZVER" ]]; then
    echo " -> Same version already present."
    read -p "    Re-download and overwrite? (y/N): " choice
    [[ "$choice" != "y" && "$choice" != "Y" ]] && echo " Skipping download." && zillaiso="$existing_iso" && goto_grub=true
  elif dpkg --compare-versions "$existing_ver" "gt" "$CZVER"; then
    echo " !! Existing version ($existing_ver) is newer than selected ($CZVER)."
    read -p "    Downgrade? (y/N): " choice
    [[ "$choice" != "y" && "$choice" != "Y" ]] && echo " Skipping downgrade." && exit 0
  else
    echo " -> Existing version ($existing_ver) is older. Updating to $CZVER."
    read -p "    Proceed with update? (y/N): " choice
    [[ "$choice" != "y" && "$choice" != "Y" ]] && echo " Skipping update." && exit 0
  fi
fi

# ==== Download ISO ====
if [[ -z "$goto_grub" ]]; then
  echo " # Downloading Clonezilla ISO (no verification)..."
  wget -O "$zillaiso" "$iso_url" || { echo " !! ISO download failed."; exit 1; }
  echo " WARNING: ISO was downloaded without verification — proceed with caution."
fi

# ==== Create GRUB Entry ====
echo " # Creating GRUB entry at /etc/grub.d/39_Clonezilla"

cat <<EOF > /etc/grub.d/39_Clonezilla
#!/bin/sh
exec tail -n +3 \$0
menuentry "Clonezilla Live ($CZVER)" --class recovery {
    search --no-floppy --fs-uuid --set $uuid
    insmod gzio
    if [ x\$grub_platform = xxen ]; then
        insmod xzio
        insmod lzopio
    fi
    insmod part_gpt
    insmod ext2
    set isofile="/iso/$zillaiso"
    loopback loop \$isofile
    linux (loop)/live/vmlinuz nomodeset boot=live live-config edd=on \\
        ocs_live_run="ocs-live-general" ocs_live_extra_param="" \\
        keyboard-layouts="$keyboard_layout" ocs_live_batch="no" \\
        locales="$current_locale" ip=frommedia toram=filesystem.squashfs \\
        findiso=\$isofile
    initrd (loop)/live/initrd.img
}
EOF

chmod +x /etc/grub.d/39_Clonezilla

# ==== Update GRUB ====
echo " # Updating GRUB..."
update-grub

echo
echo "Clonezilla ($CZVER) has been added to your boot menu."
