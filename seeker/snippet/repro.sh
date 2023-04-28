#date: 2023-04-28T16:52:00Z
#url: https://api.github.com/gists/e8bfcd41374b88e5c960e85aeeb44ea1
#owner: https://api.github.com/users/alexeldeib

#!/usr/bin/env bash
retrycmd_if_failure() {
    retries=$1; wait_sleep=$2; timeout=$3; shift && shift && shift
    for i in $(seq 1 $retries); do
        timeout $timeout "${@}" && break || \
        if [ $i -eq $retries ]; then
            echo Executed \"$@\" $i times;
            return 1
        else
            sleep $wait_sleep
        fi
    done
    echo Executed \"$@\" $i times;
}
retrycmd_if_failure_no_stats() {
    retries=$1; wait_sleep=$2; timeout=$3; shift && shift && shift
    for i in $(seq 1 $retries); do
        timeout $timeout ${@} && break || \
        if [ $i -eq $retries ]; then
            return 1
        else
            sleep $wait_sleep
        fi
    done
}

should_skip_nvidia_drivers() {
    body=$(curl -fsSL -H "Metadata: true" --noproxy "*" "http://169.254.169.254/metadata/instance?api-version=2021-02-01")
    ret=$?
    if [ "$ret" != "0" ]; then
      return $ret
    fi
    should_skip=$(echo "$body" | jq -e '.compute.tagsList | map(select(.name | test("SkipGpuDriverInstall"; "i")))[0].value // "false" | test("true"; "i")')
    echo "$should_skip" # true or false
}


should_skip_nvidia_drivers_okay() {
    body=$(curl -fsSL -H "Metadata: true" --noproxy "*" "http://169.254.169.254/metadata/instance?api-version=2021-02-01")
    ret=$?
    if [ "$ret" != "0" ]; then
      return $ret
    fi
    should_skip=$(echo "$body" | jq -e '.compute.tagsList | map(select(.name | test("AKS-MANAGED"; "i")))[0].value // "false" | test("true"; "i")')
    echo "$should_skip" # true or false
}

should_skip_nvidia_drivers_failure() {
    body=$(curl -fsSL -H "Metadata: true" --noproxy "*" "http://169.254.169.253/metadata/instance?api-version=2021-02-01")
    ret=$?
    if [ "$ret" != "0" ]; then
      return $ret
    fi
    should_skip=$(echo "$body" | jq -e '.compute.tagsList | map(select(.name | test("SkipGpuDriverInstall"; "i")))[0].value // "false" | test("true"; "i")')
    echo "$should_skip" # true or false
}

export -f should_skip_nvidia_drivers
stdout=$(retrycmd_if_failure_no_stats 1 1 1 bash -c should_skip_nvidia_drivers)
ret=$?
echo -e "stdout: '$stdout'\nreturn code: '$ret'"
export -f should_skip_nvidia_drivers_failure
stdout=$(retrycmd_if_failure_no_stats 1 1 1 bash -c should_skip_nvidia_drivers_failure)
ret=$?
echo -e "stdout: '$stdout'\nreturn code: '$ret'"
export -f should_skip_nvidia_drivers_okay
stdout=$(retrycmd_if_failure_no_stats 1 1 1 bash -c should_skip_nvidia_drivers_okay)
ret=$?
echo -e "stdout: '$stdout'\nreturn code: '$ret'"