#date: 2023-03-03T17:07:39Z
#url: https://api.github.com/gists/e0cca398919993eebc2655cb261687d6
#owner: https://api.github.com/users/zbentley

#!/usr/bin/env bash
set -xuo pipefail

brew cleanup
brew cleanup --prune=all
rm -rf "$(brew --cache)"
pip cache purge
sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder
sudo rm -rf "/Users/$USER/Library/Caches/Sublime Text/Cache/__pycache__"
find ~ -type d -name "__pycache__" -exec rm -rv {} \;
find ~ -type f -name "*.py[cod]" -exec rm -rv {} \;

bolt --clear-cache module show system
rm -rf ~/.m2
rm -rf ~/.irb_history
rm -rf ~/.cache
rm -rf ~/.kube/cache ~/.kube/http-cache
rm -rf ~/.pyenv/cache
sudo rm -rf ~/.r10k
rm -rf ~/.zcompcache ~/.zcompdump ~/.zcompdump.zwc
rm -f ~/.lesshst
rm -rf ~/Library/Caches/{pip,Keybase,black,SentryCrash,Maps,Helm,net.freemacsoft.AppCleaner,net.istumbler,pip-tools,org.wireshark.Wireshark,io.sentry}
rm -rf ~/Library/Caches/{com.plausiblelabs.crashreporter.data,com.onevcat.Kingfisher.ImageCache.default,google-sdks-events,Yarn}
rm -rf ~/Library/Caches/{corg.whispersystems.signal-desktop.ShipIt,com.spotify.client,4kdownload.com}

rm -rf ~/Library/Caches/google-sdks-events
rm -rf ~/Library/Caches/mixpanel*
rm -rf ~/Library/Caches/com.sublimetext.*
rm -rf ~/Library/Caches/com.jamf*
rm -rf ~/Library/Application Support/virtualenv
rm -rf /Users/$USER/.cache/pre-commit
docker system prune --all --force

# Commented out for fear of breaking stuff underneath running IDEs. Uncomment and run once your IDEs are shut down.
#rm -rf ~/Library/Caches/JetBrains

echo "Now's a good time to run 'Onyx' or 'Maintenance' or similar"