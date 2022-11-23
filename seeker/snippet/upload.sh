#date: 2022-11-23T17:01:02Z
#url: https://api.github.com/gists/1b8d1f71998a3111b921c919cb4b72e3
#owner: https://api.github.com/users/MichaelAllen

#!/bin/bash

# # # # # # # # # #
# This script was primarily designed to upload folders of images for a Rock powered photopass system.
# (See https://community.rockrms.com/subscriptions/rx2022/digital-photopass-powered-by-rock)
# However, it should work for uploading any type of files into any type of wokflow.
#
# Your Rock API key will need to have the following permissions at a minimum:
#   - Edit permissions for the File Type you are using
#   - Edit permissions to the Rest Controller: POST `api/Workflows/WorkflowEntry/{WorkflowTypeId}`
#
# Usage: 'upload.sh file.jpg' OR 'upload.sh path/to/directory'
# (On MacOS you can drag a folder onto the terminal to automatically fill in the directory path)
#
# If given a single file, it will upload and process that file.
# If given a directory, it will upload and process all files in that directory that match the configured extension.
# # # # # # # # # #

ROCK_URL="https://rock.valorouschurch.com"
API_KEY="REDACTED"
FILE_TYPE_GUID="db67dde1-e078-4b1b-848f-986110a804b0"
WORKFLOW_TYPE_ID="256"
EXTENSION=".jpg" # Only used when processing a directory

# # # # # # # # # #

main () {
  # Esure we have a file or directory
  local FILE_NAME=$1
  if [[ -z $FILE_NAME || ! ( -f $FILE_NAME || -d $FILE_NAME ) ]] ; then
    echo "Error. Invalid file or directory passed."
    echo "Usage: 'upload.sh photo.jpg' OR 'upload.sh path/to/directory'"
    exit -1
  fi

  PROCESSED=0
  if [ -f "${FILE_NAME}" ] ; then # We have a single file
    processfile "${FILE_NAME}"
  elif [ -d "${FILE_NAME}" ] ; then # We have a directory. Process all files inside
    echo "Processing all '${EXTENSION}' files in '${FILE_NAME}'"
    shopt -s nullglob
    for file in "${FILE_NAME}"/*$EXTENSION; do
      processfile "${file}"
    done
    shopt -u nullglob
  fi
  echo "Processed ${PROCESSED} files."
}
processfile () {
  local FILE_NAME=$1
  echo "${FILE_NAME}:"

  # Upload the file
  echo -n "    Uploading... "
  local CURL_OUT=$(mktemp)
  local CURL_STATUS=$(
    curl -s -f -H "Authorization-Token: "**********"
    "${ROCK_URL}/FileUploader.ashx?fileId=&isBinaryFile=T&fileTypeGuid=${FILE_TYPE_GUID}" \
    -F "Files=@${FILE_NAME}"
  )
  local UPLOAD_RESULT=$(<$CURL_OUT)
  rm -f $CURL_OUT
  if [ $CURL_STATUS != 200 ] ; then
    echo -e "Failed. Result: ${CURL_STATUS}\n${UPLOAD_RESULT}"
    exit -1
  else
    local REGEX='"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"'
    if [[ $UPLOAD_RESULT =~ $REGEX ]] ; then
      local FILE_GUID=${BASH_REMATCH[1]}
      echo "Done. File Guid: ${FILE_GUID}"
    else
      echo -e "Failed. Result: ${CURL_STATUS}\n${UPLOAD_RESULT}"
      exit -1
    fi
  fi

  # Fire the workflow
  echo -n "    Firing workflow... "
  local CURL_STATUS=$(
    curl -s -f -H "Authorization-Token: "**********"
    "${ROCK_URL}/api/workflows/workflowentry/${WORKFLOW_TYPE_ID}?Image=${FILE_GUID}" \
    -X POST -d {}
  )
  if [ $CURL_STATUS != 200 ] ; then
    echo -e "Failed. Result: ${CURL_STATUS}"
    exit -1
  else
    echo "Done."
  fi

  # Increment counter
  ((PROCESSED++))
}

main "$@"; exit  else
    echo "Done."
  fi

  # Increment counter
  ((PROCESSED++))
}

main "$@"; exit