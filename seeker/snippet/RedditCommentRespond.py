#date: 2021-08-31T01:39:35Z
#url: https://api.github.com/gists/00481c139854dec05e247d6e366e7b88
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
        if "scooby dooby doo" in comment.body.lower():
          comment.reply("Where are you?")