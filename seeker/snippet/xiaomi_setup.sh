#date: 2024-05-16T16:56:28Z
#url: https://api.github.com/gists/5a4480d8153a87a1b86274ff3c2b20ba
#owner: https://api.github.com/users/lokey0905

#!/bin/sh
xiaomi_security(){
  preferences=/data/data/com.miui.securitycenter/shared_prefs/remote_provider_preferences.xml
  [ ! -f "$preferences" ] && exit
  grep -q 'security_adb_install_enable' $preferences
  if [ $? -ne 0 ] ;then
    sed -i 's/<\/map>/<boolean name="security_adb_install_enable" value="true" \/>\n<\/map>/g' $preferences
  else
    sed -i '/security_adb_install_enable/ s|\([vV]alue="\)[^"]*\("\)|\1true\2|g' $preferences
  fi
  grep -q 'permcenter_install_intercept_enabled' $preferences
  if [ $? -ne 0 ] ;then
    sed -i 's/<\/map>/<boolean name="permcenter_install_intercept_enabled" value="false" \/>\n<\/map>/g' $preferences
  else
    sed -i '/permcenter_install_intercept_enabled/ s|\([vV]alue="\)[^"]*\("\)|\1false\2|g' $preferences
  fi
  am force-stop com.miui.securitycenter
  resetprop persist.security.adbinput 1
  resetprop persist.security.adbinstall 1
  resetprop --delete ro.secureboot.devicelock
}
xiaomi_security