#date: 2023-09-12T16:54:58Z
#url: https://api.github.com/gists/611f3b0bfc0de52561fab42835fe1d77
#owner: https://api.github.com/users/ermogenes

#! /bin/sh

echo "Getting version description..."
appversion=$(echo | grep  '"version" *: *"[0-9]*\.[0-9]*\.[0-9]*"' app.json | sed -r 's/[^0-9\.]//g')

if [ -z $appversion ]
then
  echo "ERROR: expo.version not found on app.json"
  exit 1
fi

echo "Bumping build number (iOS) and version code (Android)..."
oldbuildnumber=$(echo | grep  '"buildNumber" *: *"[0-9]*"' app.json | sed -r 's/[^0-9]//g')

if [ -z $oldbuildnumber ]
then
  echo "ERROR: ios.buildNumber not found on app.json"
  exit 1
fi

oldversioncode=$(echo | grep  '"versionCode" *: *[0-9]*' app.json | sed -r 's/[^0-9]//g')

if [ -z $oldversioncode ]
then
  echo "ERROR: android.versioncode not found on app.json"
  exit 1
fi

version=$((oldbuildnumber+1))
sed -ir '
  # Change iOS buildNumber
  s/"buildNumber" *: *"[0-9]*"/\"buildNumber\": \"'$version'\"/

  # Change Android versionCode
  s/"versionCode" *: *[0-9]*/\"versionCode\": '$version'/  
  ' app.json

newbuildnumber=$(echo | grep  '"buildNumber" *: *"[0-9]*"' app.json | sed -r 's/[^0-9]//g')
newversioncode=$(echo | grep  '"versionCode" *: *[0-9]*' app.json | sed -r 's/[^0-9]//g')

if [ $newbuildnumber -ne $newversioncode ]
then
  echo "ERROR: ios.buildNumber not equals to android.versionCode"
  exit 1
fi

description="v.${appversion}-$newbuildnumber"
echo "Updated from $oldbuildnumber to $newbuildnumber. Description: $description"