#date: 2021-11-24T16:54:41Z
#url: https://api.github.com/gists/1647d8740e9a433407402a3c0b5a660b
#owner: https://api.github.com/users/memes

#!/bin/sh
#
# Verify GCP service account default authentication status on VM

info()
{
  echo "$0: INFO: $*" >&2
}

error()
{
  echo "$0: ERROR: $*" >&2
  exit 1
}

command -v curl >/dev/null 2>/dev/null || error "curl not found in PATH"
command -v jq >/dev/null 2>/dev/null || error "jq not found in PATH"
code="$(curl -s -w '{"http_status": "%{http_code}"}' -o /run/sa_email -H 'Metadata-Flavor: Google' http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/email | jq -r '.http_status')"
case "${code}" in
    200)
        info "Effective service account is $(cat /run/sa_email)"
        code="$(curl -s -w '{"http_status": "%{http_code}"}' -o /run/sa_token -H 'Metadata-Flavor: Google' http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token | jq -r '.http_status')"
        info "Token endpoint Response code: ${code}"
        case "${code}" in
            200)
                info "Access token is: $(jq -r .access_token < /run/sa_token)"
                ;;
            404)
                info "An access token is not available on this VM"
                ;;
            *)
                info "Unexpected code for token endpoint: ${code}"
                ;;
        esac
        ;;
    404)
        info "This VM is running without access to a service account"
        ;;
    *)
        info "Unexpected HTTP code for email endpoint: ${code}"
        ;;
esac
[ -e /run/sa_email ] && rm -f /run/sa_email
[ -e /run/sa_token ] && rm -f /run/sa_token