#date: 2021-12-06T17:10:11Z
#url: https://api.github.com/gists/50977baeb18ba9f16213a36f9e1185d0
#owner: https://api.github.com/users/jmcerrejon

#!/bin/bash

export TITANIUM_SDK_PATH="$HOME/Library/Application Support/Titanium/mobilesdk/osx/"
export CURRENT_APPC_SDK
CURRENT_APPC_SDK=$(ls -t "$TITANIUM_SDK_PATH" | head -n 1)
export IPHONEUDID="XXX" # iPhone Simulator
export IPHONEREALID="XXX" # iPhone 11
export ANDROIDID="android" # Android 10 Google Pixel 3 - 4096 RAM - SDK 29
export ANDROIDREALID="XXX" # Real Android device

# NOTE: For add log: -l {trace, debug, info, warn or error}
#
DEV_CERT="XXX"
SDK_DIR="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs"
SIM_APP="/Applications/Xcode.app/Contents/Developer/Applications/Simulator.app/Contents/MacOS/Simulator"
alias simios="open -a Simulator"
alias updateappc="appc ti sdk install && npm -g update alloy"
alias tios="ti build --platform ios --target simulator -C $IPHONEUDID --liveview"
alias tios?="ti build --platform ios --target simulator -C ? --liveview"
alias tiosnl="ti build --platform ios --target simulator -C $IPHONEUDID"
alias tiosr="ti build --platform ios --target device --device-id '$IPHONEREALID' -V '$DEV_CERT' --skip-js-minify --liveview" #My iPhone 11
alias tiosrnl="ti build --platform ios --target device --device-id '$IPHONEREALID' -V '$DEV_CERT'"
alias iosr="ti build --platform ios -T device -C $IPHONEREALID -V $DEV_CERT --liveview --config-file --log-level info --no-banner --project-dir $HOME/Documents/Appcelerator_Studio_Workspace/foe-ahora"
alias and_emu="android -avd $ANDROIDID"
alias tand="ti build --platform android --target emulator --device-id $ANDROIDID --skip-js-minify --liveview"
alias tandnl="ti build --platform android --target emulator --skip-js-minify --device-id $ANDROIDID" # no liveview
alias tandr="ti build --platform android --target device --device-id $ANDROIDREALID --skip-js-minify --liveview"
alias tandrnl="ti build --platform android --target device --device-id $ANDROIDREALID"
alias cap_and="~/Documents/Android/sdk/platform-tools/adb shell screencap -p /sdcard/screen.png && ~/Documents/Android/sdk/platform-tools/adb pull /sdcard/screen-${CURRENT_DATE}.png"
alias cleanTesting="rm -rf ~/Library/Logs/CoreSimulator/* ~/Library/Developer/Xcode/DerivedData/*"
alias simios="$SIM_APP -SimulateApplication $SDK_DIR/iPhoneSimulator.sdk/Applications/MobileSafari.app/MobileSafari"
alias tandcb="emulator @android -no-snapshot-load"
alias tgen="tn generate" # build profiles to load with TinY. Other commands: tn list

stopliveview() {
    "$HOME/Library/Application Support/Titanium/mobilesdk/osx/$CURRENT_APPC_SDK/node_modules/liveview/bin/liveview-server" stop
    ti clean
}

clean() {
    echo -e "\nCleaning da hause...\n"
    ti clean
    echo -e "\nDeleting ./build if preceed...\n"
    [[ -d build ]] && rm -rf build
    echo -e "\nDeleting ./platform if preceed...\n"
    [[ -d platform ]] && rm -rf platform
    echo -e "\nDeleting $HOME/Library/Logs/DiagnosticReports\n"
    rm -rf /Users/ulysess/Library/Logs/DiagnosticReports/*.*
}

simboth() {
    stopliveview

    # Hack to avoid config file issues on appc run
    sleep 1

    appc run --platform ios -C "$IPHONEUDID" --quiet --liveview --liveview-eport 5556 --liveview-fport 8325 &

    # Needed to avoid on iOS Couldn't find module: localeStrings. Play with the next value knowing iOS app must be launched before Android app.
    sleep 4

    appc run --platform android -C "$ANDROIDID" --quiet --liveview
}
