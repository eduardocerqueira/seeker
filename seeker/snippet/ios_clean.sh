#date: 2021-12-27T16:42:22Z
#url: https://api.github.com/gists/0943b52e905735d4eef31fd2d3d2a3d1
#owner: https://api.github.com/users/dahabit

#!/bin/sh
echo "========== Cleanup start =========="
rm -Rf ios/Pods
rm -Rf ios/.symlink
rm -Rf ios/Flutter/Flutter.framework
rm -Rf ios/Flutter/Flutter.podspec
rm -rf ios/Podfile.lock
flutter clean
flutter packages get
cd ios
pod install
pod update
cd ..

echo "========== Cleanup end =========="
echo "Suggest you to restart your xcode"
echo "========== Have a nice day =========="
