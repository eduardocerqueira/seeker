#date: 2023-03-01T17:00:00Z
#url: https://api.github.com/gists/2ddde63e937c7b72d3f486727d6a4e44
#owner: https://api.github.com/users/gesielrosa

if [ -z "$ANDROID_SDK_ROOT" ]; then
    echo "ANDROID_SDK_ROOT is not set"
    exit 1
fi

cd $ANDROID_SDK_ROOT/emulator

echo ""
echo "[Available devices]"
./emulator -list-avds
echo ""

read -p "Enter the device id: " name

./emulator -avd $name
