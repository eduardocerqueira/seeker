#date: 2023-01-06T16:35:24Z
#url: https://api.github.com/gists/34dfc9c977e23ab98c6da066db23e782
#owner: https://api.github.com/users/mezerotm

#!/bin/bash

# Check if the script has sudo privileges
if [ "$(id -u)" -ne 0 ]; then
  # The script does not have sudo privileges
  echo "This script requires sudo privileges to run."
  echo "Please re-run the script with sudo."
  exit 1
fi

# Create a temporary file to store the hardware information
temp_file=$(mktemp)
echo "Temporary file created at: $temp_file"

# Get system information
echo "System Information:" >> $temp_file
echo "-------------------" >> $temp_file
echo "Operating System: $(uname -a)" >> $temp_file
echo "Kernel Version: $(uname -r)" >> $temp_file
echo "Machine Type: $(uname -m)" >> $temp_file
echo "" >> $temp_file

# Get CPU information
echo "CPU Information:" >> $temp_file
echo "----------------" >> $temp_file
echo "Model: $(lscpu | grep "Model name:" | awk -F: '{print $2}')" >> $temp_file
echo "Number of CPUs: $(lscpu | grep "CPU(s):" | awk -F: '{print $2}')" >> $temp_file
echo "Architecture: $(lscpu | grep "Architecture:" | awk -F: '{print $2}')" >> $temp_file
echo "CPU op-mode(s): $(lscpu | grep "CPU op-mode(s):" | awk -F: '{print $2}')" >> $temp_file
echo "Byte Order: $(lscpu | grep "Byte Order:" | awk -F: '{print $2}')" >> $temp_file
echo "" >> $temp_file

# Get memory information
echo "Memory Information:" >> $temp_file
echo "-------------------" >> $temp_file
echo "Total: $(free -h | grep "Mem:" | awk '{print $2}')" >> $temp_file
echo "Free: $(free -h | grep "Mem:" | awk '{print $4}')" >> $temp_file
echo "Used: $(free -h | grep "Mem:" | awk '{print $3}')" >> $temp_file
echo "Usage: $(free -m | grep "Mem:" | awk '{print $3/$2 * 100.0}')% (based on total memory)" >> $temp_file
echo "" >> $temp_file

# Get disk information
echo "Disk Information:" >> $temp_file
echo "-----------------" >> $temp_file
echo "$(df -h --total | grep "total")" >> $temp_file
echo "$(df -h)" >> $temp_file
echo "" >> $temp_file

# Get PCI information
echo "PCI Devices:" >> $temp_file
echo "------------" >> $temp_file
echo "$(lspci)" >> $temp_file
echo "" >> $temp_file

# Get USB information
echo "USB Devices:" >> $temp_file
echo "------------" >> $temp_file
echo "$(lsusb)" >> $temp_file
echo "" >> $temp_file

# Check if the lsscsi command is installed
if ! command -v lsscsi > /dev/null; then
  # Prompt the user to install lsscsi
  echo "The lsscsi command is not installed. Do you want to install it? (Y/N)"
  read -r install_lsscsi

  # Install lsscsi if the user says "Y" (case-insensitive)
  if [ "$(echo "$install_lsscsi" | tr '[:lower:]' '[:upper:]')" == "Y" ]; then
    sudo apt-get install lsscsi
  fi
fi

# Get SCSI information (if lsscsi is installed)
if command -v lsscsi > /dev/null; then
  echo "SCSI Devices:" >> $temp_file
  echo "-------------" >> $temp_file
  echo "$(lsscsi)" >> $temp_file
  echo "" >> $temp_file
else
  echo "SCSI Devices: Not available" >> $temp_file
  echo "" >> $temp_file
fi

# Get block device information
echo "Block Devices:" >> $temp_file
echo "--------------" >> $temp_file
echo "$(lsblk)" >> $temp_file
echo "" >> $temp_file

# Get hwinfo information
echo "Hardware Information (hwinfo):" >> $temp_file
echo "-----------------------------" >> $temp_file
echo "$(hwinfo)" >> $temp_file
echo "" >> $temp_file

# Get fdisk information
echo "Partition Information (fdisk):" >> $temp_file
echo "-----------------------------" >> $temp_file
echo "$(fdisk -l)" >> $temp_file
echo "" >> $temp_file

# Get hdparm information
echo "HDD Information (hdparm):" >> $temp_file
echo "------------------------" >> $temp_file

# Get a list of all block devices
block_devices=$(lsblk -d --output NAME)

# For each block device...
for device in $block_devices; do
# Skip the first line (which contains the column names)
  if [ "$device" == "NAME" ]; then
    continue
  fi

  # Check if the device is a hard disk (type 83)
  if [ $(fdisk -l /dev/$device | grep "Id=83" | wc -l) -gt 0 ]; then
    # Print information about the hard disk
    echo "$(hdparm -I /dev/$device)" >> $temp_file
  fi
done

echo "" >> $temp_file

# Echo the URL of the uploaded file
echo "Hardware information has been uploaded to:"

# Upload the temporary file to termbin.com
cat $temp_file | nc termbin.com 9999

# Remove the temporary file
rm $temp_file