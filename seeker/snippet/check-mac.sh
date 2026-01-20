#date: 2026-01-20T17:09:17Z
#url: https://api.github.com/gists/112e6bfd73ab59aed3cf6630803a4352
#owner: https://api.github.com/users/anotherChowdhury

#!/bin/bash

echo "--- 🖥️ MACBOOK AUTO-INSPECTION & SETUP ---"

# 1. Check/Install Homebrew
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing now (this may take a few mins)..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Setup brew for Apple Silicon path
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
else
    echo "✅ Homebrew is already installed."
fi

# 2. Check/Install smartmontools
if ! command -v smartctl &> /dev/null; then
    echo "Installing smartmontools for deep SSD check..."
    brew install smartmontools
else
    echo "✅ smartmontools is already installed."
fi

echo -e "\n--- 📊 GENERATING HEALTH REPORT ---"

echo "1. CHIP, RAM & CORES"
sysctl -n machdep.cpu.brand_string
echo "Total Cores: $(sysctl -n hw.ncpu)"
echo "Performance Cores: $(sysctl -n hw.perflevel0.logicalcpu)"
echo "Efficiency Cores: $(sysctl -n hw.perflevel1.logicalcpu)"
echo "RAM: $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024)) GB"

echo -e "\n2. GPU CORES"
# Finding the actual GPU core count
system_profiler SPDisplaysDataType | grep "Total Number of Cores" || ioreg -l | grep num_cores

echo -e "\n3. BATTERY STATUS"
system_profiler SPPowerDataType | grep -E "Cycle Count|Condition|Maximum Capacity" | sed 's/^[[:space:]]*//'

echo -e "\n4. SSD DEEP HEALTH (SMART)"
echo "Note: "**********"
sudo smartctl -a disk0 | grep -E "Percentage Used|Data Units Written|Available Spare|Power On Hours|Critical Warning"

echo -e "\n5. DISPLAY SPECS"
system_profiler SPDisplaysDataType | grep -E "Resolution|Main Display" | sed 's/^[[:space:]]*//'

echo -e "\n---------------------------------------"
echo "INSPECTION COMPLETE"-------------------"
echo "INSPECTION COMPLETE"