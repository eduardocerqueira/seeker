#date: 2025-07-25T16:49:14Z
#url: https://api.github.com/gists/978bbd261a0d0d53eb2ed493cae5cff9
#owner: https://api.github.com/users/birkin

## (server-name) DEV-SERVER code update script for '(project-name)' project -- using new uv-pyproject.toml architecture


## setup ------------------------------------------------------------
echo " "; echo "--------------------"; echo " "; echo "DEPLOY-START"; echo " "
echo ":: setting envars..."
GROUP="(the-group)"
LOG_DIR_PATH="/path/to/stuff/logs/"
PROJECT_DIR_PATH="/path/to/stuff/project/"  # for `git pull`
STATIC_MEDIA_DIR_PATH="/path/to/html/django_media/(project)_media/"
STUFF_DIR_PATH="/path/to/stuff/"
TOUCH_PATH="/path/to/stuff/project/config/tmp/restart.txt"
URL_ARRAY=( "(check-url)" )
echo "---"; echo " "; echo " "

## reset ownership and permissions 1 of 2 ---------------------------
echo ":: running initial group and permissions update..."; echo " "
path_array=( $LOG_DIR_PATH $STUFF_DIR_PATH )
for i in "${path_array[@]}"
do
  echo "processing directory: " $i
  sudo /bin/chgrp -R $GROUP $i  # recursively ensures all items are set to proper group -- solves problem of an item being root/root if sudo-updated after a forced deletion
  sudo /bin/chmod -R g=rwX $i
done
echo "---"; echo " "; echo " "

## update app -------------------------------------------------------
echo ":: running git pull..."; echo " "
cd $PROJECT_DIR_PATH
git pull
echo "---"; echo " "; echo " "

## update any python packages --------------------------------------
echo ":: running uv sync..."; echo " "
uv sync --locked --group staging  # raises an error on discrepancy with pyproject.toml
echo "---"; echo " "; echo " "

## run collectstatic ------------------------------------------------
echo ":: running collectstatic..."
uv run ./manage.py collectstatic --noinput
echo "---"; echo " "; echo " "

## reset group and permissions 2 of 2 -------------------------------
echo ":: running cleanup group and permissions update..."; echo " "
path_array=( $STUFF_DIR_PATH $STATIC_MEDIA_DIR_PATH )
for i in "${path_array[@]}"
do
  echo "processing directory: " $i
  sudo /bin/chgrp -R $GROUP $i  # recursively ensures all items are set to proper group -- solves problem of an item being root/root if sudo-updated after a forced deletion
  sudo /bin/chmod -R g=rwX $i
done
echo "---"; echo " "; echo " "

## make it real -----------------------------------------------------
echo ":: touching the restart file..."
touch $TOUCH_PATH
#sleep 1
echo "---"; echo " "; echo " "

## run tests --------------------------------------------------------
echo ":: VERIFICATION 1 of 2; running tests..."; echo " "
uv run ./run_tests.py
if [ $? -ne 0 ]; then
  echo "TESTS FAILED!"
else
  echo "tests passed!"
fi
echo "---"; echo " "; echo " "

## check urls -------------------------------------------------------
echo ":: VERIFICATION 2 of 2; performing curl-check..."
for i in "${URL_ARRAY[@]}"
do
 echo " "; echo "checking url: " $i
 RESPONSE=$( curl --head --silent --max-time 3 $i )
 #echo "$RESPONSE"
 if [[ $RESPONSE == *"HTTP/1.1 200 OK"* ]]; then
   echo "curl-check: good!"
 else
     echo "curl-check: PROBLEM -- no 200?"
 fi
done
echo "---"; echo " "; echo " "


echo "DEPLOY-COMPLETE"; echo " "; echo "--------------------"; echo " "

## [END]
