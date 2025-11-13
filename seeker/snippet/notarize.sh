#date: 2025-11-13T16:59:57Z
#url: https://api.github.com/gists/fcd988d0253c51104e8bbe050e057891
#owner: https://api.github.com/users/mykulyak

#!/usr/bin/env bash

TEAM_ID="..."
APPLE_ID="..."
APP_SPECIFIC_PASSWORD= "**********"

RAW_JSON=$(xcrun notarytool submit "notarization-bundle.zip" \
    --apple-id "$APPLE_ID" \
    --team-id "$TEAM_ID" \
    --password "$APP_SPECIFIC_PASSWORD" \
    --output-format json)
echo "$RAW_JSON"
SUBMISSION_ID=$(echo "$RAW_JSON" | jq -r '.id')
echo "Submission ID: $SUBMISSION_ID"

while true; do
    STATUS=$(xcrun notarytool info "$SUBMISSION_ID" \
        --apple-id "$APPLE_ID" \
        --team-id "$TEAM_ID" \
        --password "$APP_SPECIFIC_PASSWORD" \
        --output-format json | jq -r '.status')

    case $STATUS in
        "Accepted")
            echo -e "Notarization succeeded!"
            break
            ;;
        "In Progress")
            echo "Notarization in progress... waiting 30 seconds"
            sleep 30
            ;;
        "Invalid"|"Rejected")
            echo "Notarization failed with status: $STATUS"
            xcrun notarytool log "$SUBMISSION_ID" \
                --apple-id "$APPLE_ID" \
                --team-id "$TEAM_ID" \
                --password "$APP_SPECIFIC_PASSWORD"
            exit 1
            ;;
        *)
            echo "Unknown status: $STATUS"
            exit 1
            ;;
    esac
done