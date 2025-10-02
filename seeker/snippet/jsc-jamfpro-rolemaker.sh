#date: 2025-10-01T17:12:40Z
#url: https://api.github.com/gists/d6a9a231e78eb6254c3c711e8b51dc93
#owner: https://api.github.com/users/wmehilos

#!/bin/zsh --no-rcs

##### JSON VARS #####
deviceLifecycleJSON='{
"privileges": [
	"Read Smart Computer Groups",
	"Read Mobile Devices",
	"Read Smart Mobile Device Groups",
	"Read Static Mobile Device Groups",
	"Create Static Computer Groups",
	"Read Computers",
	"Read Mobile Device Applications",
	"Read Static Computer Groups",
	"Create Static Mobile Device Groups",
	"Read Mac Applications"
],
"displayName": "Device Lifecycle Management"
}'
deviceRiskJSON='{
	"privileges": [
		"Update Computer Extension Attributes",
		"Update Mobile Devices",
		"Read Mobile Device Extension Attributes",
		"Update Computers",
		"Update Mobile Device Extension Attributes",
		"Update User",
		"Delete Computer Extension Attributes",
		"Read Computer Extension Attributes",
		"Delete Mobile Device Extension Attributes",
		"Create Computer Extension Attributes",
		"Create Mobile Device Extension Attributes"
	],
	"displayName": "Device Risk UEM Signaling"
}'
configProfileDeployJSON='{
	"privileges": [
		"Read macOS Configuration Profiles",
		"Read iOS Configuration Profiles",
		"Create iOS Configuration Profiles",
		"Update Smart Mobile Device Groups",
		"Update Static Computer Groups",
		"Update iOS Configuration Profiles",
		"Create macOS Configuration Profiles",
		"Update Smart Computer Groups",
		"Update Static Mobile Device Groups",
		"Update macOS Configuration Profiles"
	],
	"displayName": "Configuration Profile Deployment"
}'

clientDeployJSON='{
	"authorizationScopes": [
		"Device Lifecycle Management",
		"Configuration Profile Deployment",
		"Device Risk UEM Signaling"
	],
	"displayName": "UEM-Connect",
	"enabled": true,
	"accessTokenLifetimeSeconds": "**********"
}'

function lock_lb () { ingress=$(curl -s -u "$username": "**********": //g') }

function get_bearer_token() {
	response=$(curl -s -b "$ingress" -u "$username": "**********"
	bearerToken= "**********"
	tokenExpiration= "**********"
	tokenExpirationEpoch= "**********"
}

function check_token_expiration() {
	current_epoch=$(date +%s)
	if [[ $token_expiration_epoch -ge $current_epoch ]]; then
		echo "Token valid until the following epoch time: "**********"
	else
		get_bearer_token 
	fi
}

function invalidate_token() {
	responseCode=$(curl -w "%{http_code}" -H "Authorization: "**********"
	if [[ ${responseCode} == 204 ]]; then
		echo "Token successfully invalidated"
		access_token= "**********"
		token_expiration_epoch= "**********"
	elif [[ ${responseCode} == 401 ]]; then
		:
	else
		echo "An unknown error occurred invalidating the token"
	fi
}

function get_url_and_creds () {
	echo -n "Enter your Jamf Pro Server URL: https://"; read url
	url="https://$url"
	echo -n "Enter your username: "; read username
	echo -n "Enter $username's password: "**********"
}



function create_device_lifecycle () {
	curl -s -b "$ingress" -H "Authorization: "**********": application/json" -X POST "$url/api/v1/api-roles" -d $deviceLifecycleJSON --fail-with-body
	echo
}

function create_device_risk () {
	curl -s -b "$ingress" -H "Authorization: "**********": application/json" -X POST "$url/api/v1/api-roles" -d $deviceRiskJSON --fail-with-body
	echo
}

function create_config_deploy () {
	curl -s -b "$ingress" -H "Authorization: "**********": application/json" -X POST "$url/api/v1/api-roles" -d $configProfileDeployJSON --fail-with-body
	echo
}

function create_client () {
	curl -s -b "$ingress" -H "Authorization: "**********": application/json" -X POST "$url/api/v1/api-integrations" -d $clientDeployJSON --fail-with-body
	echo
}

function main() {
	# Setup
	get_url_and_creds
	lock_lb
	check_token_expiration
	# Create Roles
	create_device_lifecycle 
	create_config_deploy
	create_device_risk
	# Create Client
	create_client
	# Cleanup
	invalidate_token
}	
main "$@"$bearerToken" -H "Content-type: "**********"
	echo
}

function main() {
	# Setup
	get_url_and_creds
	lock_lb
	check_token_expiration
	# Create Roles
	create_device_lifecycle 
	create_config_deploy
	create_device_risk
	# Create Client
	create_client
	# Cleanup
	invalidate_token
}	
main "$@"ice_risk
	# Create Client
	create_client
	# Cleanup
	invalidate_token
}	
main "$@"