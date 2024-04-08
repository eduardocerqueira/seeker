#date: 2024-04-08T17:09:04Z
#url: https://api.github.com/gists/78e1bb3536a4100bf82ba7e9df71f181
#owner: https://api.github.com/users/thee-thonifho-muhali

#!/data/data/com.termux/files/usr/bin/bash

am start -a android.settings.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS -d package:com.termux

echo -ne "\n Checking if Termux has storage permission..."
rm -r ~/storage >/dev/null 2>&1
if ! touch /storage/emulated/0/.tmp_check_termux >/dev/null 2>&1
then
    echo -e "\nGrant Termux storage permission and run the script again\n"
    termux-setup-storage
    exit 1
fi
rm -rf /storage/emulated/0/.tmp_check_termux >/dev/null 2>&1
termux-setup-storage
echo "done"

echo -e "\n Grant Termux Display over other apps permission"

sleep 5

am start -a android.settings.action.MANAGE_OVERLAY_PERMISSION -d package:com.termux

yes | pkg upgrade -y

yes | pkg install -y tur-repo x11-repo git python-pip

yes | pkg install -y chromium

pkg clean

apt autoremove

pip install -U selenium undetected-chromedriver git+https://github.com/ugorsahin/ChatGPT_Automation

chat_automation_dir="$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())" 2>/dev/null)/chatgpt_automation"

sed -i '/out = subprocess/d' "${chat_automation_dir}/helpers.py"

sed -i "s/out = re.*/out = re.search\(r\'\(\\d{3}\)\', \"110\"\)/" "${chat_automation_dir}/helpers.py"

sed -i 's#driver_executable_path=driver_executable_path,#driver_executable_path="/data/data/com.termux/files/usr/bin/chromedriver",#' "${chat_automation_dir}/chatgpt_client.py"

value="true"; key="allow-external-apps"; file="/data/data/com.termux/files/home/.termux/termux.properties"; mkdir -p "$(dirname "$file")"; chmod 700 "$(dirname "$file")"; if ! grep -E '^'"$key"'=.*' $file &>/dev/null; then [[ -s "$file" && ! -z "$(tail -c 1 "$file")" ]] && newline=$'\n' || newline=""; echo "$newline$key=$value" >> "$file"; else sed -i'' -E 's/^'"$key"'=.*/'"$key=$value"'/' $file; fi