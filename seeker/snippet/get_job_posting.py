#date: 2021-10-20T17:07:45Z
#url: https://api.github.com/gists/14140575c6ddd1daa2a89621a11506ba
#owner: https://api.github.com/users/haayanau

import time
import pandas as pd
import numpy as np
from newspaper import Article, ArticleException, Config


url = "https://neuvoo.com/view/?jpos=&jp=1&l=&id=5cc0b117a2cf&lang=en&sal-job=1&ss=1&context=serp&testid=champion&nb=true&reqid=05412eb2337b92dbca9c2ea510dc2053&source=neuvoosearch"

def download_single_article(link):
    practice = False
    # link is a single url from a list of links
    article_url = link
    article_dict = {}
    article_dict["link"] = article_url
    # create Article object
    article = Article(article_url, fetch_images=False, verbose=True) # language=lang_code
    # Download contents of article object
    if not practice:
        article.download()
        # not all articles can be parsed
        try:
            article.parse()
            article_dict["text"] = article.text
            article_dict["title"] = article.title
            article_dict["authors"] = article.authors
            article_dict["date"] = article.publish_date

        except ArticleException:
            article_dict["text"] = np.nan
            article_dict["title"] = np.nan
            article_dict["date"] = np.nan
            article_dict["authors"] = np.nan

    else:
        # this is for practice! :-)
        time.sleep(1)
    return article_dict

x = download_single_article(url)

print(x)