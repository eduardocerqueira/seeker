#date: 2023-08-29T16:54:37Z
#url: https://api.github.com/gists/4ac79cf504324845c5525c441133b0f5
#owner: https://api.github.com/users/zhiyue

{ 
 echo "Loaded kernel extensions:";
 kextstat -kl | awk '!/com\.apple/{printf "%s %s\n", $6, $7}';
 echo $'\n'"Loaded user agents:";
 launchctl list | sed 1d | awk '!/0x|com\.apple|org\.(x|openbsd)|\.[0-9]+$/{print $3}';
 echo $'\n'"Inserted libraries:";
 launchctl getenv DYLD_INSERT_LIBRARIES;
 echo $'\n'"User cron tasks:";
 crontab -l;
 echo $'\n'"System launchd configuration:";
 cat /e*/lau*;
 echo $'\n'"User launchd configuration:";
 cat .lau*;
 echo $'\n'"Login items:";
 osascript -e 'tell application "System Events" to get name of login items';
 echo $'\n'"Extrinsic loadable bundles:";
 cd;
 find -L /S*/L*/E* {,/}L*/{Ad,Compon,Ex,In,Keyb,Mail/Bu,P*P,Qu,Scripti,Servi,Spo}* -type d -name Contents -prune | while read d;
 do /usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$d/Info.plist" | egrep -qv "^com\.apple\.[^x]|Accusys|ArcMSR|ATTO|HDPro|HighPoint|driver\.stex|hp-fax|JMicron|print|SoftRAID" && echo ${d%/Contents};
 done;
# echo $'\n'"Unsigned shared libraries:";
# find /u*/{,*/}lib -type f -exec sh -c 'file -b $1 | grep -qw shared && ! codesign -v $1' {} {} \;
 -print;
 echo;
 ls -A {,/}L*/{Launch,Priv,Sta}*;
 } 2> /dev/null
 
{
echo "Loaded system agents:"; 
sudo launchctl list | sed 1d | awk '!/0x|com\.(apple|openssh|vix\.cron)|org\.(amav|apac|cups|isc|ntp|postf|x)/{print $3}';
echo $'\n'"Login hook:";
sudo defaults read com.apple.loginwindow LoginHook;
echo $'\n'"Root cron tasks:";
sudo crontab -l;
echo $'\n'"Log check:";
syslog -k Sender kernel -k Message CReq 'GPU |hfs: Ru|I/O e|find tok|n Cause: -|NVDA\(|pagin|timed? ?o' | tail;
} 2> /dev/null