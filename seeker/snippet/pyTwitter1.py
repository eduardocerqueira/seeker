#date: 2022-10-19T17:25:58Z
#url: https://api.github.com/gists/5097c334b6e55393190a04bb27266d35
#owner: https://api.github.com/users/code-and-dogs

import tweepy
import credentials

api_key = credentials.API_KEY
api_key_secret = "**********"
bearer_token = "**********"
access_token = "**********"
access_token_secret = "**********"

auth = "**********"
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

tweetMessage='Hello World from Python Tweepy'
api.update_status(tweetMessage).API(auth)

tweetMessage='Hello World from Python Tweepy'
api.update_status(tweetMessage)