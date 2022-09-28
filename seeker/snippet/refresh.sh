#date: 2022-09-28T17:23:39Z
#url: https://api.github.com/gists/891be9a8af7d0c46c46078f5295246cf
#owner: https://api.github.com/users/sean-e-dietrich

#!/usr/bin/env bash

#: exec_target = cli

## Refresh database, files, and import configuration
##
## Usage: fin refresh

# Abort if anything fails
set -e

LIVE=false
ELEMENTS='db'

PANTHEON_SITE=explo
PANTHEON_ENV=dev

# options may be followed by one colon to indicate they have a required argument
if ! options=$(getopt -o s:e:d:l -l site:,env:,live,data: -- "$@")
then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

while [ $# -gt 0 ]
do
    case $1 in
    -s|--site) PANTHEON_SITE="$2"; shift ;;
    -e|--env) PANTHEON_ENV="$2"; shift ;;
    -l|--live) LIVE=true; ;;
    -d|--data) ELEMENTS="$2"; shift;;
    esac
    shift
done

green='\033[0;32m'
yellow='\033[1;33m'
NC='\033[0m'

divider='===================================================\n'
check='\xE2\x9C\x85'
construction='\xF0\x9F\x9A\xA7'
crossmark='\xE2\x9D\x8C'
hospital='\xF0\x9F\x8F\xA5'
party='\xF0\x9F\x8E\x88 \xF0\x9F\x8E\x89 \xF0\x9F\x8E\x8A'
reverseparty='\xF0\x9F\x8E\x8A \xF0\x9F\x8E\x89 \xF0\x9F\x8E\x88'
rocket='\xF0\x9F\x9A\x80'
silhouette='\xF0\x9F\x91\xA4'
lightning='\xE2\x9A\xA1'
drop='\xF0\x9F\x92\xA7'
shark='\xF0\x9F\xA6\x88'
gear='\xEF\xB8\x8F'

DOCROOT=${DOCROOT:-docroot};
SITE_DIRECTORY=${SITE_DIRECTORY:-default};

DOCROOT_PATH="${PROJECT_ROOT}/${DOCROOT}"
SITEDIR_PATH="${DOCROOT_PATH}/sites/${SITE_DIRECTORY}"

cd $PROJECT_ROOT
cd $SITEDIR_PATH

if [ ! -d ~/tmp ]; then
    mkdir ~/tmp
fi

if [ $ELEMENTS = 'all' ] || [ $ELEMENTS = 'db' ]; then
    DBFILE="/tmp/${PANTHEON_ENV}.${PANTHEON_SITE}.sql"
    if [ ! -f $DBFILE ] || [ ! -z $(find ${DBFILE} -mmin +360) ]; then
     echo "${DBFILE} need updating"

      echo "Exporting latest database..."
      if [ -f $DBFILE ] && [ ! -z $(find ${DBFILE} -mmin +360) ]; then
        rm -rf $DBFILE
      fi

      if $LIVE ; then
        terminus env:wake ${PANTHEON_SITE}.${PANTHEON_ENV}
        DBCONN=$(terminus connection:info ${PANTHEON_SITE}.${PANTHEON_ENV} --field="MySQL Command")
        DBDUMP=${DBCONN/mysql /mysqldump }
        eval $DBDUMP " --result-file=${DBFILE}"
      else
        COUNT=$(terminus backup:list ${PANTHEON_SITE}.${PANTHEON_ENV} --element=db --format=list | wc -l)
        if [ $COUNT -eq 1 ]; then
          echo "Creating backup on Pantheon..."
          terminus backup:create ${PANTHEON_SITE}.${PANTHEON_ENV} --element=db
        fi

        LASTBKUP=$(terminus backup:list ${PANTHEON_SITE}.${PANTHEON_ENV} --element=db --field=date | head -n 1)
        LASTBKUP=$(printf "%.0f" ${LASTBKUP})
        LASTBKUP=$(date --date="@${LASTBKUP}" "+%Y-%m-%d %H:%M:%S")
        YESTERDAY=$(date --date='yesterday' "+%Y-%m-%d %H:%M:%S")

        if [ "${LASTBKUP}" \< "${YESTERDAY}" ]; then
          echo "Old backup on Pantheon...Creating new one..."
          terminus backup:create ${PANTHEON_SITE}.${PANTHEON_ENV} --element=db
        fi

        terminus backup:get ${PANTHEON_SITE}.${PANTHEON_ENV} --element="db" --to="${DBFILE}.gz"
        gunzip ${DBFILE}.gz
      fi
    fi

    if [[ "$(${PROJECT_ROOT}/bin/drupal 2>/dev/null)" =~ "database:drop" ]]; then
        echo "Dropping Old Database..."
        ${PROJECT_ROOT}/bin/drupal database:drop -y
    fi

    echo "Importing Database..."
    drush sql-cli < ${DBFILE}
fi

if [ $ELEMENTS = 'all' ] || [ $ELEMENTS = 'files' ]; then
    cd $SITEDIR_PATH
    echo "Downloading latest set of files from ${PANTHEON_SITE}..."

    FILES_DIRECTORY=${PROJECT_ROOT}/sites/${SITE_DIRECTORY}/files
    mkdir ${FILES_DIRECTORY}
    if $LIVE ; then
        if [ ! -d ${FILES_DIRECTORY} ]; then
            mkdir ${FILES_DIRECTORY}
        fi
        cd $FILES_DIRECTORY
      terminus rsync ${PANTHEON_SITE}.${PANTHEON_ENV}:files .
    else
      FILES="/tmp/${PANTHEON_ENV}.${PANTHEON_SITE}.tar.gz"
      terminus backup:get ${PANTHEON_ENV}.${PANTHEON_SITE} --element=files --to="${FILES}"
      rm -rf ${FILES_DIRECTORY}
      tar -xf ${FILES}
      mv files_* files
    fi
    echo "Fixing files directory permissions..."
    chmod -R 755 files
fi

cd ${DOCROOT_PATH}

echo -e "\n${yellow} ${crossmark} Clearing cache ${crossmark}${NC}"
echo -e "${green}${divider}${NC}"
cd ${DOCROOT_PATH}
drush cr -y

echo -e "\n${yellow} ${drop} Updating Drupal Database ${drop}${NC}"
echo -e "${green}${divider}${NC}"
drush updatedb -y

echo -e "\n${yellow} ${crossmark} Clearing cache again ${crossmark}${NC}"
echo -e "${green}${divider}${NC}"
drush cr -y

echo -e "\n${yellow} ${gear} Database refreshed!!! ${gear}${NC}"
echo -e "${green}${divider}${NC}"

echo "Done!"