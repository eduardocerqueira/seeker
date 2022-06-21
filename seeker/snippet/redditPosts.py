#date: 2022-06-21T17:06:32Z
#url: https://api.github.com/gists/2cf8a96220f0f59b32196c54a6e089c7
#owner: https://api.github.com/users/AndrewCrider

subredditName = 'writingprompts'

def getRedditPosts(subredditName):
    wpcsv_columns = ['summary','style', 'genre' ,'title', 'url', 'id', 'author']
    writing_prompts_dict = []
    wpCSV = locationPath+ "/wprompts500.csv"

    session = Session()
    reddit = redditInfo
    
    subreddit = reddit.subreddit(subredditName)
    top_subredditlist = subreddit.top(limit=500)

    for submission in top_subredditlist:
        #Adding Columns for User Summarization and Decisions
        array = ['','','',submission.title, submission.url, submission.id, submission.author]
        writing_prompts_dict.append(dict(zip(wpcsv_columns, array)))

    print(len(writing_prompts_dict))

    with open(wpCSV, "w") as csv_file:
        writer = csv.DictWriter(csv_file, wpcsv_columns)
        writer.writeheader()
        writer.writerows(writing_prompts_dict)
    csv_file.close()