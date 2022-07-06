#date: 2022-07-06T17:13:16Z
#url: https://api.github.com/gists/63d6d6e812de442a43360367b76eafcb
#owner: https://api.github.com/users/jazlopez

# =============================================================================================
# Bash custom function and profile autocomplete to list available zones for a given profile.
# Jaziel Lopez jazlopez @ github.com
# 2022
# =============================================================================================

# --
# list aws profiles from default aws credentials file
# --
function _completion_aws_profile_names_(){
  grep -Eo "\[.*\]" $HOME/.aws/credentials | sed 's/[][]//g'
}

# --
# list available zonez
# --
function _list_available_zones_() {

	clear
	echo "[INFO] usage: list-available-zones [aws_profile]"
	echo "\tif argument [aws_profile] not provided it uses default"
	echo "========================================================================"
	local profile="default"

	[[ ! -z $1 ]] && profile=$1

	echo "[INFO] list AWS available zones for profile $profile"
	aws ec2 describe-availability-zones --profile "${profile}" | jq -r '.AvailabilityZones[].ZoneName'
}

complete -F _completion_aws_profile_names_ _list_available_zones_

alias list-available-zones=_list_available_zones_
