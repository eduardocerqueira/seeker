#date: 2022-04-13T17:17:26Z
#url: https://api.github.com/gists/cc6213726cd0a641be7022dd0179dca0
#owner: https://api.github.com/users/mubingchen

#!/bin/bash

FILEPATH="$(readlink -f $0)"
BASEDIR="$(dirname $FILEPATH)"

sdate=$1
#edate=$2
edate=`date -d "${sdate}  1 days "  "+%Y-%m-%d"`

echo $sdate"===>"$edate

$BASEDIR/mysql_dump -h=9.2.xx.xx -P=9xxx \
-u=gongyiw -p=xxxx -n=gongyi_99 \
-t=t_201999_matching_log -fields=f_txid \
-batch=10000 -cond="f_dtime>='$sdate' and f_dtime<'$edate' " \
-incrfield=f_id -incr=30000000 | cut -f2 | sort >$sdate"_99.log"

./mysql_dump -h=100.95.xx.xx -P=3xxx \
-u=gongyi_warehouse -p=xxxx -n=gongyi_warehouse \
-t=t_project_donation_log_2019 -fields=f_txid -batch=10000 \
-cond="f_dtime>='$sdate' and f_dtime<'$edate' " -incrfield=f_id \
-incr=30000000 | cut -f2 | sort >$sdate"_trans.log"

diff $sdate"_99.log" $sdate"_trans.log" | grep '>' > "repair_"${sdate}".log"


 ./mysql_dump -h=100.94.206.138 -P=3635 -u=gongyi -p=gY@2017 -n=gongyi_wx -t=t_wx_transcode -fields=f_transcode -batch=10000 -cond="f_donate_time>='2019-02-24' and f_donate_time<'2019-02-25' and f_actid=210540" -incrfield=f_id -incr=152225400 | cut -f2 | sort > real_trans.log
