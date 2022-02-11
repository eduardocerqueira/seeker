#date: 2022-02-11T16:46:05Z
#url: https://api.github.com/gists/3be19e87e25cbd876062ae34b7e232ca
#owner: https://api.github.com/users/EmilRamsvik

import tweepy
def authenticate_twitter() -> tweepy.API:
    """Generates an api object that can be used to post tweets.

    Returns:
        tweepy.API: Authenticated twitter api object.
    """
    auth = tweepy.OAuthHandler(
        settings.TWITTER_CONSUMER_KEY, settings.TWITTER_CONSUMER_SECRET
    )
    auth.set_access_token(
        settings.TWITTER_ACCESS_TOKEN, settings.TWITTER_ACCESS_TOKEN_SECRET
    )
    return tweepy.API(auth)
