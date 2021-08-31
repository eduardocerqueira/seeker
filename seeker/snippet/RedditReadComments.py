#date: 2021-08-31T01:36:13Z
#url: https://api.github.com/gists/e52b8963ba5b2dd3274a0792417a5a8f
#owner: https://api.github.com/users/JakenHerman

  import os
  import praw
  
  reddit = praw.Reddit(
        username = os.environ.get('REDDIT_USERNAME'),
        password = os.environ.get('REDDIT_PASSWORD'),
	    client_id = os.environ.get('API_CLIENT'),
	    client_secret = os.environ.get('API_SECRET'),
	    user_agent = "Scooby Searcher Bot"
  )

    for comment in reddit.subreddit('cartoons').comments(limit=1000):
        print(comment.body)