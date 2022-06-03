#date: 2022-06-03T16:46:16Z
#url: https://api.github.com/gists/2e5ce5f7f0c48589ff81ac871ebc5dbe
#owner: https://api.github.com/users/meleu

#!/usr/bin/env bash

# ... muito conteúdo foi omitido aqui...

# functions to interact with gitlab's API
###############################################################################
# https://docs.gitlab.com/ee/api/access_requests.html#valid-access-levels
declare -Ar GITLAB_ROLES=(
  [guest]=10
  [reporter]=20
  [developer]=30
  [maintainer]=40
  [owner]=50
)

gitlabApi() {
  local path="$1"
  curl \
    --silent \
    --header "PRIVATE-TOKEN: ${READ_API_TOKEN}" \
    "${CI_API_V4_URL}/${path}"
}

getUserAccessLevel() {
  gitlabApi \
    "projects/${CI_PROJECT_ID}/members/all/${GITLAB_USER_ID}" \
    | jq --exit-status '.access_level'
}

maintainersOnly() {
  [[ "$(getUserAccessLevel)" -eq "${GITLAB_ROLES[maintainer]}" ]] \
    || msgBannerError "Only Maintainers of this repository can run this job"
}
