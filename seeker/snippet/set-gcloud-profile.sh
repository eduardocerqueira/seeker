#date: 2025-10-09T16:32:22Z
#url: https://api.github.com/gists/0c0f6053a00b781d704b4d9e86284c21
#owner: https://api.github.com/users/xPlorinRolyPoly

#!/bin/bash

PROFILE_NAME="developer"

echo "Setting up gcloud profile: $PROFILE_NAME"

ACTIVE_PROFILE=$(gcloud config configurations list --filter="is_active:true" --format="value(name)")
if [ "$ACTIVE_PROFILE" = "$PROFILE_NAME" ]; then
    echo "Profile $PROFILE_NAME is active. Login complete."
else
    echo "Profile $PROFILE_NAME is not active. Current active profile: $ACTIVE_PROFILE"
    if gcloud config configurations list --format="get(name)" | grep -q "$PROFILE_NAME"; then
        echo "Profile $PROFILE_NAME already exists."
        echo "Activating existing profile: $PROFILE_NAME"
        gcloud config configurations activate "$PROFILE_NAME"
        echo "Profile $PROFILE_NAME is active. Login complete."
    else
        gcloud config configurations create "$PROFILE_NAME"
        echo "Profile $PROFILE_NAME is active."
        echo "Please authenticate with your Google account using command:"
        echo "gcloud auth login"

    fi
fi