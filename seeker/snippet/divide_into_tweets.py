#date: 2022-02-11T17:07:07Z
#url: https://api.github.com/gists/d33858ddb87280f75a9fb950e9e1f3ab
#owner: https://api.github.com/users/EmilRamsvik

def divide_into_tweet(text: str) -> List[str]:
    """Takes a textstring and divides it unto a list of strings that are less
    than or equal to 276 characters long.

    Args:
        text (str): text to be divided into tweets

    Returns:
        List[str]: list of tweets less than 276
    """
    puncts = [".", ",", ";", "--"]
    tweets = []
    while len(text) > 280:
        cut_where, cut_why = max((text.rfind(punc, 0, 276), punc) for punc in puncts)
        if cut_where <= 0:
            cut_where = text.rfind(" ", 0, 276)
            cut_why = " "
        cut_where += len(cut_why)
        tweets.append(text[:cut_where].rstrip())
        text = text[cut_where:].lstrip()
    tweets.append(text)
    return tweets