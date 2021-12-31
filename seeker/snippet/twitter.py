#date: 2021-12-31T16:38:48Z
#url: https://api.github.com/gists/829130a40ff2f8faca8ede5bee6856cd
#owner: https://api.github.com/users/janvaneck1994

import tweepy

API_KEY = ''
API_SECRET_KEY = ''

auth = tweepy.AppAuthHandler(API_KEY, API_SECRET_KEY)
api = tweepy.API(auth)

username = 'BoredApeYC'
user = api.get_user(screen_name=username)
followers_count = user.followers_count

print(f"{username} has {followers_count} followers")