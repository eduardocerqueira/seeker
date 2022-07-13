#date: 2022-07-13T17:08:25Z
#url: https://api.github.com/gists/2187566a9d1245ea2f7e36ec61efb8c4
#owner: https://api.github.com/users/lbussell

#!/bin/bash
set -euxo pipefail

# Example: ./rebuild.sh msbuild

semaphores=(
    "CreateBuildOutputProps.complete",
    "CreateCombinedRestoreSourceAndVersionProps.complete",
    "UpdateNuGetConfig.complete",
    "CopyPackage.complete",
    "Build.complete",
)

semaphore_path="artifacts/obj/semaphores/$1/"

for semaphore in "${semaphores}"
do
    rm -rf "${semaphore_path}${semaphore}"
done

rm -rf "src/$1/artifacts/source-build/self/package-cache/*"

./build.sh --online

cat "src/$1/artifacts/source-build/self/prebuilt-report/annotated-usage.xml"