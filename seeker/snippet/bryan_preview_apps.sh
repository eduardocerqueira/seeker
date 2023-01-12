#date: 2023-01-12T17:02:34Z
#url: https://api.github.com/gists/242a9b8bebe7651f62837513b3680c2f
#owner: https://api.github.com/users/brillantewang

########## Instructions ##########
#1. Create the apps you are using on heroku.com - mapp, manage, dapi, dapi-tasks
#2. Copy over variables to "Modification Section". Leave them blank/commented if you're not using them
#3. Run this script


########## Modification Section - Start ##########
# Configurations
SHOULD_CREATE_DATABASE=false

# Apps - Fill in Review App Names that were created
MAPP_APP=""
MANAGE_APP=""
DAPI_APP=""
DAPI_TASKS_APP=""
########## Modification Section - End ##########


#################### DO NOT MODIFY ####################
DEFAULT_MAPP_APP="facility-manage-livestage"
DEFAULT_MANAGE_APP="facility-manage-livestage"
DEFAULT_DAPI_APP="secure-dapi-livestage-2"
DEFAULT_DAPI_TASKS_APP="facility-batch-livestage"

DEFAULT_DAPI_DYNO_SIZE="worker=1:Standard-2X"
DEFAULT_DAPI_TASKS_DYNO_SIZE="worker=2:Standard-2X"

MAPP_APP_TO_USE=$DEFAULT_MAPP_APP
MANAGE_APP_TO_USE=$DEFAULT_MANAGE_APP
DAPI_APP_TO_USE=$DEFAULT_DAPI_APP
DAPI_TASKS_APP_TO_USE=$DEFAULT_DAPI_TASKS_APP

if [ -n "${MAPP_APP}" ]; then
  MAPP_APP_TO_USE=$MAPP_APP
fi
if [ -n "${MANAGE_APP}" ]; then
  MANAGE_APP_TO_USE=$MANAGE_APP
fi
if [ -n "${DAPI_APP}" ]; then
  DAPI_APP_TO_USE=$DAPI_APP
fi
if [ -n "${DAPI_TASKS_APP}" ]; then
  DAPI_TASKS_APP_TO_USE=$DAPI_TASKS_APP
fi

### Frontend
## mapp
echo "#################### mapp ####################"
if [ -n "${MAPP_APP}" ]; then
  echo "mapp: setting config vars"

  echo "mapp: setting dapi to $DAPI_APP_TO_USE"
  heroku config:set -a $MAPP_APP DAPI_CACTUS_HOST=https://$DAPI_APP_TO_USE.herokuapp.com/
  heroku config:set -a $MAPP_APP DAPI_HOST=https://$DAPI_APP_TO_USE.herokuapp.com/
  heroku config:set -a $MAPP_APP DAPI_HOST_NO_CACHE=https://$DAPI_APP_TO_USE.herokuapp.com/
else
  echo "mapp: app not included, skipping"
fi

## manage
echo "#################### manage ####################"
if [ -n "${MANAGE_APP}" ]; then
  echo "manage: setting MAPP_HOST to $DAPI_APP_TO_USE"
  heroku config:set -a $MANAGE_APP MAPP_HOST=https://$MAPP_APP_TO_USE.herokuapp.com/

  echo "manage: setting MANAGE_HOST to $MANAGE_APP_TO_USE"
  heroku config:set -a $MANAGE_APP MANAGE_HOST=https://$MANAGE_APP_TO_USE.herokuapp.com/

  echo "manage: setting dapi to $DAPI_APP_TO_USE"
  heroku config:set -a $MANAGE_APP DAPI_CACTUS_HOST=https://$DAPI_APP_TO_USE.herokuapp.com/
  heroku config:set -a $MANAGE_APP DAPI_CLOUDFRONT_HOST=https://$DAPI_APP_TO_USE.herokuapp.com/
  heroku config:set -a $MANAGE_APP DAPI_HOST=https://$DAPI_APP_TO_USE.herokuapp.com/
else
  echo "manage: app not included, skipping"
fi


### Backend
## dapi
echo "#################### dapi ####################"
if [ -n "${DAPI_APP}" ]; then
  echo "dapi: updating dynos"
  heroku dyno:scale -a $DAPI_APP $DEFAULT_DAPI_DYNO_SIZE
  
  ## Database
  if [ "$SHOULD_CREATE_DATABASE" = true ]; then
    echo 'Creating fork of livestage and attaching to dapi'
    heroku addons:create heroku-postgresql:standard-2 --fork $DEFAULT_DAPI_APP::DATABASE_URL -a $DAPI_APP --as=DATABASE

    echo "dapi: attaching database $DAPI_APP::DATABASE"
    heroku addons:attach -a $DAPI_APP $DAPI_APP::DATABASE --confirm $DAPI_APP --as=READ_ONLY_DATABASE
    heroku addons:attach -a $DAPI_APP $DAPI_APP::DATABASE --confirm $DAPI_APP --as=SLOW_QUERY_DATABASE
  fi

  if [ -n "${DAPI_TASKS_APP}"]; then
    echo "dapi: creating redis"
    heroku addons:create -a $DAPI_APP heroku-redis:premium-2
  else
    echo "dapi: attaching redis $DAPI_TASKS_APP_TO_USE::REDIS"
    heroku addons:attach -a $DAPI_APP --confirm $DAPI_APP $DAPI_TASKS_APP_TO_USE::REDIS --as=REDIS
  fi
else
  echo "dapi: app not included, skipping"
fi

## dapi-tasks
echo "#################### dapi-tasks ####################"
if [ -n "${DAPI_TASKS_APP}" ]; then
  echo "dapi-tasks: updating dynos"
  heroku dyno:scale -a $DAPI_TASKS_APP $DEFAULT_DAPI_TASKS_DYNO_SIZE

  if [ -n "${DAPI__APP}"]; then
    echo "dapi-tasks: attaching database $DAPI_APP_TO_USE::DATABASE"
    heroku addons:attach -a $DAPI_TASKS_APP --confirm $DAPI_TASKS_APP $DAPI_APP_TO_USE::DATABASE --as=DATABASE
    heroku addons:attach -a $DAPI_TASKS_APP --confirm $DAPI_TASKS_APP $DAPI_APP_TO_USE::DATABASE --as=DATABASE_POOL
    heroku addons:attach -a $DAPI_TASKS_APP --confirm $DAPI_TASKS_APP $DAPI_APP_TO_USE::DATABASE --as=PROD_FOLLOWER_DATABASE_POOL
    heroku addons:attach -a $DAPI_TASKS_APP --confirm $DAPI_TASKS_APP $DAPI_APP_TO_USE::DATABASE --as=PROD_FOLLOWER_DATABASE

    echo "dapi-tasks: attaching redis $DAPI_APP_TO_USE::REDIS"
    heroku addons:attach -a $DAPI_TASKS_APP --confirm $DAPI_TASKS_APP $DAPI_APP_TO_USE::REDIS --as=REDIS
  else
    echo "dapi-tasks: ERROR - NO DAPI PROVIDED, use case does not make sense"
  fi
else
  echo "dapi-tasks: app not included, skipping"
fi


### Database: Creation Status
if [ -n "${DAPI_APP}" ] && [ "$SHOULD_CREATE_DATABASE" = true ]; then
  heroku pg:wait $DAPI_APP::DATABASE  -a $DAPI_APP
fi