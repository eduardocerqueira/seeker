#date: 2021-08-31T01:31:59Z
#url: https://api.github.com/gists/b67e02aaf5a72b9e768d3e0c81064ff2
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