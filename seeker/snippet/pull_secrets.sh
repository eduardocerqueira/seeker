#date: 2023-11-10T16:42:05Z
#url: https://api.github.com/gists/d890c9a24ac2c0f68162c901562511d0
#owner: https://api.github.com/users/marcinantkiewicz

#! /usr/bin/env sh
set -o pipefail

FILEPATH=$1; shift;

function pull_secrets {
	MANIFEST=$1; shift;
	SECRETS= "**********"=" + .versionName');

	PROJECT_ID=$(gcloud projects list --filter $(gcloud config get project) --format="value(PROJECT_NUMBER)")

	for SECRET in ${SECRETS[@]}; do
		 SECRET_ENV= "**********"=' -f 1);
		SECRET_PATH= "**********"=' -f 2);
		SECRET_NAME= "**********"
		echo "export ${SECRET_ENV}= "**********"=$SECRET_NAME --project=$PROJECT_ID)\"";
	done
}

set -e
test -r "$FILEPATH" -a -f "$FILEPATH" || \
	(>&2 echo "Error: file \"$FILEPATH\" not found or unreadable"; exit 255);

pull_secrets "$FILEPATH";

set -e
test -r "$FILEPATH" -a -f "$FILEPATH" || \
	(>&2 echo "Error: file \"$FILEPATH\" not found or unreadable"; exit 255);

pull_secrets "$FILEPATH";