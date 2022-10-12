#date: 2022-10-12T17:07:08Z
#url: https://api.github.com/gists/851c98774c77b64f88f72fad533906d8
#owner: https://api.github.com/users/yavuzKomecoglu

def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    
    #Clean only digits
    tweet = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", tweet)
    
    # Replaces URLs with the word URL
    #tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)
    
    # Replace @handle with the word USER_MENTION
    #tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    tweet = re.sub(r'@[\S]+', '', tweet)
    
    # Replaces #hashtag with hashtag
    #tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    tweet = re.sub(r'#(\S+)', '', tweet)
    
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')

    # Replace emojis with either EMO_POS or EMO_NEG
    #tweet = handle_emojis(tweet)
    tweet = remove_emoji(tweet)
   
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)

    #my custom chars
    tweet = tweet.replace('₺','')
    tweet = tweet.replace('=','')
    tweet = tweet.replace('’','')
    tweet = tweet.replace('|','')
    tweet = tweet.replace('‘','')
    tweet = tweet.replace('/','')
    tweet = tweet.replace('…','')
    tweet = tweet.replace('–','')
    tweet = tweet.replace('&','')
    tweet = tweet.replace('“','')
    tweet = tweet.replace('”','')
    tweet = tweet.replace('+','')
    tweet = tweet.replace('%','')
    tweet = tweet.replace('@','')
    tweet = tweet.replace('#','')

    words = "**********"

    for word in words:
      word = preprocess_word(word)
      #if is_valid_word(word):
      #    processed_tweet.append(word)
      processed_tweet.append(word)

    return ' '.join(processed_tweet)'.join(processed_tweet)