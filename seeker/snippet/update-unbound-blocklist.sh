#date: 2022-07-26T17:10:33Z
#url: https://api.github.com/gists/ac85187242cb49d0e212d0862fb0f794
#owner: https://api.github.com/users/funasoul

#!/usr/bin/env sh

## @since 2020-05-15 
## @last 2021-01-05 
## @author takuya
## unbound にブロッキング
## ブロッキングするURLを定期的に更新する

function update_domain_lists(){
  update280blocker
  updateYoutubeAdblocker
}
function geneate_conf(){

  URL=$1
  OUTPUT=$2
  curl -sL $URL \
    | sed -e "s/\r//"   \
    | grep -E '^[a-z0-9]' \
    | awk '{print "local-zone: \""$1".\" static"}' \
    | sort \
    | uniq \
    > $OUTPUT
}

function checkHTTPStatusIs200(){
  URL=$1
  RET=$(curl -IL -s -o /dev/null -w '%{http_code}' $URL)

  [[ $RET == 200 ]];

}
function update280blocker () {

  URL=https://280blocker.net/files/280blocker_domain_$(date +%Y%m).txt
  ## mirror
  ## URL=https://raw.githubusercontent.com/junkurihara/280blocker_domain-mirror/master/280blocker_domain.txt
  OUTPUT=/etc/unbound/280blocker-list.conf
  if ! checkHTTPStatusIs200 $URL ; then
    echo failed;
    return 1
  fi
  ##
  geneate_conf $URL $OUTPUT
}

function updateYoutubeAdblocker () {

  URL=https://raw.githubusercontent.com/anudeepND/youtubeadsblacklist/master/domainlist.txt
  OUTPUT=/etc/unbound/youtube-ad.conf
  ##
  if ! checkHTTPStatusIs200 $URL ; then
    echo failed;
    return 1
  fi
  ##
  geneate_conf $URL $OUTPUT
}

function restart_unbund(){
  [ -e /etc/init.d/unbound ] &&
  /etc/init.d/unbound restart
}
function restart_dnsmasq(){
  [ -e /etc/init.d/dnsmasq ] &&
  /etc/init.d/dnsmasq restart 2>&1 > /dev/null
}
function main(){
  update_domain_lists &&
  restart_unbund &&
  restart_dnsmasq
}



main

