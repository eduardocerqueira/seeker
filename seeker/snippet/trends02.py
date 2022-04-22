#date: 2022-04-22T17:17:43Z
#url: https://api.github.com/gists/dc94dfe4216198a8e74a7efc850dd9fb
#owner: https://api.github.com/users/advaithhl

import logging
import os

import tweepy

# Initialise logging to basic config.
logging.basicConfig()

try:
    token = os.environ['TRENDAGRAM_TWITTER_BEARER_TOKEN']
    auth = tweepy.OAuth2BearerHandler(token)
    api = tweepy.API(auth)
except KeyError:
    logging.error(
        'The environment variable for the Twitter Bearer Token was not found. Please make sure that an the Twitter Bearer Token is provided as an environment variable named "TRENDAGRAM_TWITTER_BEARER_TOKEN".')
    exit()