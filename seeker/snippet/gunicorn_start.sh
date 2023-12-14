#date: 2023-12-14T16:45:57Z
#url: https://api.github.com/gists/cf23e8dfc7831544921078dbb5183771
#owner: https://api.github.com/users/Taisho

#!/bin/bash
# Purpose: Gunicorn starter
# Author: manojit.gautam@gmail.com
# Name of an application
NAME="Your projectname"
# project directory
PROJECTDIR=/webapps/example.com
# django project virutalenv directory
VENVDIR=/webapps/example.com/venv
# Project source directory
SRCDIR=/webapps/example.com/master/src
# Sock file as gunicorn will communicate using unix socket
SOCKFILE=$PROJECTDIR/gunicorn.sock
# User who runs the app
USER=webapps
# the group to run as
GROUP=webapps
# how many worker processes should Gunicorn spawn
NUM_WORKERS=3
# which settings file should Django use
# If you haven't spit your file it should example.settings only
DJANGO_SETTINGS_MODULE=example.settings.production
# WSGI module name
DJANGO_WSGI_MODULE=example.wsgi
# Activate the virtual environment
source $VENVDIR/bin/activate
# Export the settings module
export DJANGO_SETTINGS_MODULE=$DJANGO_SETTINGS_MODULE
# Export the python path from virtualenv dir
export PYTHONPATH=$DJANGODIR:$PYTHONPATH
# move to src dir !IMPORTANT otherwise it won't work.
cd $SRCDIR
# Start your Django Unicorn
# Programs meant to be run under supervisor should not daemonize themselves (do not use --daemon)
exec $VENVDIR/bin/gunicorn ${DJANGO_WSGI_MODULE}:application \
  --name $NAME \
  --workers $NUM_WORKERS \
  --user=$USER --group=$GROUP \
  --bind=unix:$SOCKFILE \
  --log-level=debug \
  --log-file=-
