#date: 2023-10-10T17:09:06Z
#url: https://api.github.com/gists/e3e18386e33cfc16b9d02f2661676881
#owner: https://api.github.com/users/spookyuser

#! /bin/bash

# Manual stuff
# Install gcloud sdk: https://cloud.google.com/sdk/docs/install
# Install firebase tools: https://firebase.google.com/docs/cli#install_the_firebase_cli
# Login to firebase: firebase login
# Login to gcloud: gcloud auth login

GCLOUD_PROJECT_ID=xxxxx
BACKUP_BUCKET=backups-$GCLOUD_PROJECT_ID-firebase

gcloud config set project $GCLOUD_PROJECT_ID
gcloud bucket create $BACKUP_BUCKET
gcloud beta firestore export --async $BACKUP_BUCKET

firebase use $GCLOUD_PROJECT_ID
firebase auth:export users.json --format=JSON

# More manual stuff
# Wait a few min then download the bucket, it takes like 5 min to export:
# - gsutil -m cp -r gs://$BACKUP_BUCKET .
#
# - Save the Password Paramaters from the firebase authentication section https: "**********"
gle.com/docs/cli/auth#password_hash_parameters in your password manager probably
