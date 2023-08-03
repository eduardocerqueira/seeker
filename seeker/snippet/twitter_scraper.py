#date: 2023-08-03T17:04:08Z
#url: https://api.github.com/gists/1cd581fc33248be05e0b90462751430b
#owner: https://api.github.com/users/cmj

import csv
import json
import requests
import argparse
import datetime
import time
import re

# All values stored here are constant, copy-pasted from the website
FEATURES_USER = '{"blue_business_profile_image_shape_enabled":true,"responsive_web_graphql_exclude_directive_enabled":true,"verified_phone_label_enabled":false,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,"responsive_web_graphql_timeline_navigation_enabled":true}'

#FEATURES_TWEETS = '{"blue_business_profile_image_shape_enabled":true,"responsive_web_graphql_exclude_directive_enabled":true,"verified_phone_label_enabled":false,"responsive_web_graphql_timeline_navigation_enabled":true,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,"tweetypie_unmention_optimization_enabled":true,"vibe_api_enabled":true,"responsive_web_edit_tweet_api_enabled":true,"graphql_is_translatable_rweb_tweet_is_translatable_enabled":true,"view_counts_everywhere_api_enabled":true,"longform_notetweets_consumption_enabled":true,"tweet_awards_web_tipping_enabled":false,"freedom_of_speech_not_reach_fetch_enabled":true,"standardized_nudges_misinfo":true,"tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":false,"interactive_text_enabled":true,"responsive_web_text_conversations_enabled":false,"longform_notetweets_rich_text_read_enabled":true,"responsive_web_enhance_cards_enabled":false}'
FEATURES_TWEETS = '{"rweb_lists_timeline_redesign_enabled":false,"responsive_web_graphql_exclude_directive_enabled":true,"verified_phone_label_enabled":false,"creator_subscriptions_tweet_preview_api_enabled":true,"responsive_web_graphql_timeline_navigation_enabled":true,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,"tweetypie_unmention_optimization_enabled":true,"responsive_web_edit_tweet_api_enabled":true,"graphql_is_translatable_rweb_tweet_is_translatable_enabled":true,"view_counts_everywhere_api_enabled":true,"longform_notetweets_consumption_enabled":true,"responsive_web_twitter_article_tweet_consumption_enabled":false,"tweet_awards_web_tipping_enabled":false,"freedom_of_speech_not_reach_fetch_enabled":true,"standardized_nudges_misinfo":true,"tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":true,"longform_notetweets_rich_text_read_enabled":true,"longform_notetweets_inline_media_enabled":true,"responsive_web_media_download_video_enabled":false,"responsive_web_enhance_cards_enabled":false}'

AUTHORIZATION_TOKEN = "**********"
HEADERS = { 'authorization': "**********"
        # The Bearer value is a fixed value that is copy-pasted from the website
        # 'x-guest-token': "**********"
}

GET_USER_URL = 'https://twitter.com/i/api/graphql/sLVLhk0bGj3MVFEKTdax1w/UserByScreenName'
#GET_TWEETS_URL = 'https://twitter.com/i/api/graphql/CdG2Vuc1v6F5JyEngGpxVw/UserTweets'
GET_TWEETS_URL = 'https://twitter.com/i/api/graphql/XicnWRbyQ3WgVY__VataBQ/UserTweets'
FIELDNAMES = ['id', 'tweet_url', 'name', 'user_id', 'username', 'published_at', 'content', 'views_count', 'retweet_count', 'likes', 'quote_count', 'reply_count', 'bookmarks_count', 'medias']

class TwitterScraper:

    def __init__(self, username):
        # We do initiate requests Session, and we get the `guest-token` from the HomePage
        resp = requests.get("https://twitter.com/")
        self.gt = resp.cookies.get_dict().get("gt") or "".join(re.findall(r'(?<=\"gt\=)[^;]+', resp.text))
        assert self.gt
        HEADERS['x-guest-token'] = "**********"
        # assert self.guest_token
        self.HEADERS = HEADERS
        assert username
        self.username = username

    def get_user(self):
        # We recover the user_id required to go ahead
        arg = {"screen_name": self.username, "withSafetyModeUserFields": True, "includePromotedContent": True, "withQuickPromoteEligibilityTweetFields": True, "withVoice": True, "withV2Timeline": True}
        
        params = {
            'variables': json.dumps(arg),
            'features': FEATURES_USER,
        }

        response = requests.get(
            GET_USER_URL,
            params=params, 
            headers=self.HEADERS
        )

        try: 
            json_response = response.json()
        except requests.exceptions.JSONDecodeError: 
            print(response.status_code)
            print(response.text)
            raise

        result = json_response.get("data", {}).get("user", {}).get("result", {})
        legacy = result.get("legacy", {})

        return {
        "id": result.get("rest_id"), 
        "username": self.username, 
        "full_name": legacy.get("name")
        }

    def tweet_parser(
    	self,
        user_id, 
        full_name, 
        tweet_id, 
        item_result, 
        legacy
        ):

        # It's a static method to parse from a tweet
        medias = legacy.get("entities").get("media")
        medias = ", ".join(["%s (%s)" % (d.get("media_url_https"), d.get('type')) for d in legacy.get("entities").get("media")]) if medias else None

        return {
            "id": tweet_id,
            "tweet_url": f"https://twitter.com/{self.username}/status/{tweet_id}",
            "name": full_name,
            "user_id": user_id,
            "username": self.username,
            "published_at": legacy.get("created_at"),
            "content": legacy.get("full_text"),
            "views_count": item_result.get("views", {}).get("count"),
            "retweet_count": legacy.get("retweet_count"),
            "likes": legacy.get("favorite_count"),
            "quote_count": legacy.get("quote_count"),
            "reply_count": legacy.get("reply_count"),
            "bookmarks_count": legacy.get("bookmark_count"),
            "medias": medias
        }

    def iter_tweets(self, limit=20):
        # The main navigation method
        print(f"[+] scraping: {self.username}")
        _user = self.get_user()
        full_name = _user.get("full_name")
        user_id = _user.get("id")
        if not user_id:
            print("/!\\ error: no user id found")
            raise NotImplementedError
        cursor = None
        _tweets = []

        while True:
            var = {
            "userId": user_id, 
            "count": 20, 
            "cursor": cursor, 
            "includePromotedContent": True,
            "withDownvotePerspective": False,
            "withReactionsMetadata": False,
            "withReactionsPerspective": False,
            "withQuickPromoteEligibilityTweetFields": True, 
            "withVoice": False,
            "withV2Timeline": True
            }

            params = {
                'variables': json.dumps(var),
                'features': FEATURES_TWEETS,
            }

            response = requests.get(
                GET_TWEETS_URL,
                params=params,
                headers=self.HEADERS,
            )

            json_response = response.json()
            #XXX
            print(json_response)
            result = json_response.get("data", {}).get("user", {}).get("result", {})
            timeline = result.get("timeline_v2", {}).get("timeline", {}).get("instructions", {})
            entries = [x.get("entries") for x in timeline if x.get("type") == "TimelineAddEntries"]
            entries = entries[0] if entries else []

            for entry in entries:
                content = entry.get("content")
                entry_type = content.get("entryType")
                tweet_id = entry.get("sortIndex")
                if entry_type == "TimelineTimelineItem":
                    item_result = content.get("itemContent", {}).get("tweet_results", {}).get("result", {})
                    legacy = item_result.get("legacy")

                    if legacy.get("full_text"):
                        tweet_data = self.tweet_parser(user_id, full_name, tweet_id, item_result, legacy)
                        _tweets.append(tweet_data)

                if entry_type == "TimelineTimelineCursor" and content.get("cursorType") == "Bottom":
                    cursor = content.get("value")

                if len(_tweets) >= limit:
                    # We do stop — once reached tweets limit provided by user
                    break

            print(f"[#] tweets scraped: {len(_tweets)}")


            if len(_tweets) >= limit or cursor is None or len(entries) == 2:
                break

        return _tweets

    def generate_csv(self, tweets=[]):
        import datetime
        timestamp = int(datetime.datetime.now().timestamp())
        filename = '%s_%s.csv' % (self.username, timestamp)
        print('[+] writing %s' % filename)
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter='\t')
            writer.writeheader()

            for tweet in tweets: 
                print(tweet['id'], tweet['published_at'])
                writer.writerow(tweet)


def main():
    print('start')
    s = time.perf_counter()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--username', '-u', type=str, required=False, help='user to scrape tweets from', default='elonmusk')
    argparser.add_argument('--limit', '-l', type=int, required=False, help='max tweets to scrape', default=100)
    args = argparser.parse_args()
    username = args.username
    limit = args.limit

    assert all([username, limit])

    twitter_scraper = TwitterScraper(username)
    tweets = twitter_scraper.iter_tweets(limit=limit)
    assert tweets
    twitter_scraper.generate_csv(tweets)
    print('elapsed %s' % (time.perf_counter()-s))
    print('''~~ success
 _       _         _            
| |     | |       | |           
| | ___ | |__  ___| |_ __ __  
| |/ _ \| '_ \/ __| __/| '__|
| | (_) | |_) \__ \ |_ | |   
|_|\___/|_.__/|___/\__||_|   
''')


if __name__ == '__main__':
    main()
 __/| '__|
| | (_) | |_) \__ \ |_ | |   
|_|\___/|_.__/|___/\__||_|   
''')


if __name__ == '__main__':
    main()
