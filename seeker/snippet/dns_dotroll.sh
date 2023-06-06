#date: 2023-06-06T16:55:09Z
#url: https://api.github.com/gists/0f78f81afbed91b6235b8a7c4eed1681
#owner: https://api.github.com/users/gnanet

#!/usr/bin/bash

# Dotroll domain api
# - api access values stored per domain, including migration from account stored values
# - zone data is POST-ed to avoid "414 Request-URI Too Large" errors
#
# Initially export values Dotroll_User and Dotroll_Password
# export Dotroll_User= "**********"='<dotroll_api_password>'; acme.sh --issue --dns dns_dotroll -d <domain.tld> -d '*.<domain.tld>'

# Usage: add  _acme-challenge.www.domain.com   "XKrxpRBosdIKFzxW_CT3KLZNf6q0HG9i01zxXp5CPBs"
dns_dotroll_add() {
  fulldomain=$1
  txtvalue=$2

  _info "Adding TXT record using dotroll API"
  _debug fulldomain "$fulldomain"
  _debug txtvalue "$txtvalue"

  Dotroll_User="${Dotroll_User:-$(_readdomainconf Dotroll_User)}"
  Dotroll_Password="${Dotroll_Password: "**********"
  if [ -z "$Dotroll_User" ] || [ -z "$Dotroll_Password" ]; then
    _debug "Try to migrate from account conf"
    Acc_Dotroll_User="$(_readaccountconf_mutable Dotroll_User)"
    Acc_Dotroll_Password= "**********"
    if [ -n "$Acc_Dotroll_User" ] && [ -n "$Acc_Dotroll_Password" ]; then
      Dotroll_User="$Acc_Dotroll_User"
      Dotroll_Password= "**********"
      _savedomainconf Dotroll_User "$Acc_Dotroll_User"
      _savedomainconf Dotroll_Password "$Acc_Dotroll_Password"
    else
      Dotroll_User=""
      Dotroll_Password= "**********"
      _err "You don't specify dotroll user accounts."
      _err "Please create you key and try again."
      return 1
    fi
  fi

  #save the user and password to the domain conf file.
  _savedomainconf Dotroll_User "$Dotroll_User"
  _savedomainconf Dotroll_Password "$Dotroll_Password"

  _debug "First detect the root zone"
  if ! _get_root "$fulldomain"; then
    _err "invalid domain"
    return 1
  fi

  _debug _sub_domain "$_sub_domain"
  _debug _domain "$_domain"

  _debug "Getting existing records"

  _dotroll_rest GET "$_domain/get"
  if ! _contains "$response" 'message":"OK"'; then
    return 1
  fi

  if _contains "$response" "$txtvalue"; then
    _info "The record is existing, skip"
    return 0
  fi

  # IMPORTANT this does only add new records, the original records are not altered, but are required by the API
  records=$(echo "$response" | cut -c 12- | rev | cut -c 40- | rev)
  records_list=$(printf "{%s}" "$records")

  # unchanged records_list
  modify=$(echo "$records_list" | _url_encode)
  new=$(echo '[{"name": "'${fulldomain}'.", "type": "TXT", "ttl": 3600, "txtdata": "'${txtvalue}'"}]' | _url_encode)

  _dotroll_rest POST "$_domain/modify" "modify=${modify}&new=${new}"
  if ! _contains "$response" 'message":"OK"'; then
    _err "Add txt record error."
    return 1
  fi

  _info "Added, sleeping 10 seconds"
  _sleep 10
  #todo: check if the record takes effect
  return 0
}


# Usage: rm  _acme-challenge.www.domain.com   "XKrxpRBosdIKFzxW_CT3KLZNf6q0HG9i01zxXp5CPBs"
dns_dotroll_rm() {
  fulldomain=$1
  txtvalue=$2

  _info "Removing TXT record using dotroll API"
  _debug fulldomain "$fulldomain"
  _debug txtvalue "$txtvalue"

  Dotroll_User="${Dotroll_User:-$(_readdomainconf Dotroll_User)}"
  Dotroll_Password="${Dotroll_Password: "**********"
  if [ -z "$Dotroll_User" ] || [ -z "$Dotroll_Password" ]; then
    _debug "Try to migrate from account conf"
    Acc_Dotroll_User="$(_readaccountconf_mutable Dotroll_User)"
    Acc_Dotroll_Password= "**********"
    if [ -n "$Acc_Dotroll_User" ] && [ -n "$Acc_Dotroll_Password" ]; then
      Dotroll_User="$Acc_Dotroll_User"
      Dotroll_Password= "**********"
      _savedomainconf Dotroll_User "$Acc_Dotroll_User"
      _savedomainconf Dotroll_Password "$Acc_Dotroll_Password"
    else
      Dotroll_User=""
      Dotroll_Password= "**********"
      _err "You don't specify dotroll user accounts."
      _err "Please create you key and try again."
      return 1
    fi
  fi

  #save the user and password to the domain conf file.
  _savedomainconf Dotroll_User "$Dotroll_User"
  _savedomainconf Dotroll_Password "$Dotroll_Password"

  _debug "First detect the root zone"
  if ! _get_root "$fulldomain"; then
    _err "invalid domain"
    return 1
  fi

  _debug _sub_domain "$_sub_domain"
  _debug _domain "$_domain"

  _debug "Getting existing records"

  _dotroll_rest GET "$_domain/get"
  if ! _contains "$response" 'message":"OK"'; then
    return 1
  fi

  if ! _contains "$response" "$txtvalue"; then
    _info "The TXT record is missing, skip"
    return 0
  fi

  # IMPORTANT this does only remove a single record by pairs of fulldomain + txtvalue
  records=$(echo "$response" | cut -c 12- | rev | cut -c 40- | rev)
  records_list=$(printf "{%s}" "$records")

  record=$(echo $records_list | _egrep_o ",\"[0-9]+\"\:{*\"name\":\"$fulldomain\.\"[^}]*\"${txtvalue}\"}")
  record_cnt=$(echo $records_list | _egrep_o ",\"[0-9]+\"\:{*\"name\":\"$fulldomain\.\"[^}]*\"${txtvalue}\"}" | wc -l)
  if [ ! -z "${record}" ]; then
    if [ $record_cnt -gt 1 ]; then
     while read recordline
     do
        records_list=$(echo "${records_list}" | sed -e "s|${recordline}||")
        _debug _records_list "${records_list}"
      done < <(echo "${record}")
    else
      records_list=$(echo "${records_list}" | sed -e "s|${record}||")
      _debug _records_list "${records_list}"
    fi
  fi

  # modified records_list
  modify=$(echo "$records_list" | _url_encode)

  # URL-param 'modify' only!
  _dotroll_rest POST "$_domain/modify" "modify=${modify}"
  if ! _contains "$response" 'message":"OK"'; then
    _err "Remove txt record error."
    return 1
  fi

  _info "Removed, sleeping 10 seconds"
  _sleep 10
  #todo: check if the record takes effect
  return 0
}

#_acme-challenge.www.domain.com
#returns
# _sub_domain=_acme-challenge.www
# _domain=domain.com
_get_root() {
  domain=$1
  i=2
  p=1
  while true; do
    h=$(printf "%s" "$domain" | cut -d . -f $i-100)
    if [ -z "$h" ]; then
      #not valid
      return 1
    fi

    _dotroll_rest GET "$h/get"
    if ! _contains "$response" 'message":"OK"'; then
      _debug "$h not found or error"
    else
      _sub_domain=$(printf "%s" "$domain" | cut -d . -f 1-$p)
      _domain="$h"
      return 0
    fi
    p="$i"
    i=$(_math "$i" + 1)
  done
  return 1
}

_dotroll_rest() {
  m="$1"
  ep="$2"
  data="$3"
  _debug m $m
  _debug ep $ep

  _dotroll_auth=$(printf "%s: "**********"

  export _H1="Authorization: Basic $_dotroll_auth"

  #response=$(_get "https://api.dotroll.com/domains/zone/$ep")

  if [ "$m" != "GET" ]; then
    _debug data "$data"
    response="$(_post "$data" "https://api.dotroll.com/domains/zone/$ep" "" "$m")"
  else
    response=$(_get "https://api.dotroll.com/domains/zone/$ep")
  fi

  if [ "$?" != "0" ]; then
    _err "error $ep"
    return 1
  fi

  _debug response "$response"

  return 0
}lse
    response=$(_get "https://api.dotroll.com/domains/zone/$ep")
  fi

  if [ "$?" != "0" ]; then
    _err "error $ep"
    return 1
  fi

  _debug response "$response"

  return 0
}