#date: 2022-03-03T17:04:56Z
#url: https://api.github.com/gists/d15efd17d7311ff81eae2b50d8f72f48
#owner: https://api.github.com/users/guparan

#! /bin/sh

echo "Hello World, from an external shell script!"
if [ "$BUILD_ENV" = "demo" ]; then
  echo "This is a demo."
elif [ "$BUILD_ENV" ]; then
  echo "BUILD_ENV=$BUILD_ENV"
else
  echo "There isn't a BUILD_ENV variable set."
fi
if [ "$SPAM_STRING" ]; then echo "Did you know that $SPAM_STRING?"; fi
