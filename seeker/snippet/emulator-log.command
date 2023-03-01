#date: 2023-03-01T17:02:07Z
#url: https://api.github.com/gists/fa0606fbb7598f4536d182e42ee04d72
#owner: https://api.github.com/users/gesielrosa

read -p "Enter the log filter: " filter
echo ""
echo "[Logging to '$filter']"
echo ""

adb logcat | grep $filter
