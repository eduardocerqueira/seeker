#date: 2022-11-14T17:08:17Z
#url: https://api.github.com/gists/4edf9d7abd83d8fe923b65292d5bc3dd
#owner: https://api.github.com/users/patrickdrouin

import newspaper
from newspaper import Config
from newspaper import Article

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'

config = Config()
config.browser_user_agent = USER_AGENT
config.request_timeout = 10

#base_url = 'http://www.euronews.com'
#base_url = 'http://www.cnn.com'
base_url = 'http://www.foxnews.com/'
article_urls = set()
euronews = newspaper.build(base_url, config=config, memoize_articles=False, language='en')
for sub_article in euronews.articles:
   if sub_article.url not in article_urls:
     article_urls.add(sub_article.url)
     article = Article(sub_article.url, config=config, memoize_articles=False, language='en')
     article.download()
     article.parse()

     # The majority of the article elements are located
     # within the meta data section of the page's
     # navigational structure
     article_meta_data = article.meta_data

     published_date = {value for (key, value) in article_meta_data.items() if key == 'date.created'}
     article_published_date = " ".join(str(x) for x in published_date)

     article_title = article.title

     summary = {value for (key, value) in article_meta_data.items() if key == 'description'}
     article_summary = " ".join(str(x) for x in summary)

     keywords = ''.join({value for (key, value) in article_meta_data.items() if key == 'keywords'})
     keywords_list = sorted(keywords.lower().split(','))
     article_keywords = ', '.join(keywords_list).strip()

     # the replace is used to remove newlines
     article_text = article.text.replace('\n', '')
     print(article_text)
