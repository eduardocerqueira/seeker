#date: 2024-05-06T16:50:53Z
#url: https://api.github.com/gists/313b1bc4997a7b832a29727593c90dfe
#owner: https://api.github.com/users/wheresjames

#!/bin/bash

echo "Cleaning Xcode..."

# Clear Xcode Caches
CLEANDIRS=("~/Library/Developer/Xcode/DerivedData" \
          "~/Library/Caches/org.swift.swiftpm" \
          "~/Library/Caches/org.swift.swiftpm.$USER" \
          "$(getconf DARWIN_USER_CACHE_DIR)/org.llvm.clang/ModuleCache" \
          "$(getconf DARWIN_USER_CACHE_DIR)/org.llvm.clang.$USER/ModuleCache" \
          "~/Library/Developer/Xcode/iOS DeviceSupport" \
          "~/Library/Developer/Xcode/iOS DeviceSupport Logs" \
          "~/Library/Developer/Xcode/macOS DeviceSupport" \
          "~/Library/Developer/CoreSimulator/Devices/" \
          "~/Library/Developer/CoreSimulator/Caches" \
          "~/Library/Caches/com.apple.dt.Instruments" \
          "~/Library/Caches/com.apple.dt.Xcode" \
          "~/Library/Caches/com.apple.dt.Xcode.sourcecontrol.Git")

for dir in "${CLEANDIRS[@]}"; do
    full=$(eval echo $dir)
    if [ -d "$full" ]; then
        echo "[x] $full..."
        rm -rf $full
    else
        echo "[ ] $full"
    fi
done
