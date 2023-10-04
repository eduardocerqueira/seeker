#date: 2023-10-04T17:02:00Z
#url: https://api.github.com/gists/46bad9a19d7f095e9dbd6e0effe324fe
#owner: https://api.github.com/users/ArrayOfLilly

from bs4 import BeautifulSoup
import lxml
import requests

# with open("website.html") as file:
#     contents = file.read()
#     # print(contents)
#
# soup = BeautifulSoup(contents, "lxml")
# print(soup.title)
# print(soup.title.name)
# print(soup)
# print(soup.prettify())
# only the first element with requested tag
# print(soup.p)

# print(soup.findAll('a'))
# all_anchor_tags = soup.findAll(name='a')
# print(all_anchor_tags)

# heading = soup.find(name='h1', id='name')
# print(heading)

# by CSS selector
# print(soup.select_one("#name"))
# print(soup.select(".heading"))
# print(soup.select("p a"))

response = requests.get('https://news.ycombinator.com/news')
# print(response.text)

yc_web_page = response.text
soup = BeautifulSoup(yc_web_page, "lxml")
# article_title = soup.select_one('span.titleline a').getText()
# article_link = soup.select_one('span.titleline a').get('href')
# article_upvote = soup.select_one('span.score').getText()
# print(article_title)
# print(article_link)
# print(article_upvote)

article_titles = []
article_links = []
articles = soup.select('span.titleline a')
for article_tag in articles:
    text = article_tag.getText()
    article_titles.append(text)
    link = article_tag.get('href')
    article_links.append(link)

article_upvotes = [int(score.getText().split(' ')[0]) for score in soup.select('span.score')]

print(article_titles)
print(article_links)
print(article_upvotes)

largest_number = max(article_upvotes)
print(largest_number)
most_popular_index = article_upvotes.index(largest_number)
print(most_popular_index)
print(article_titles[most_popular_index])
print(article_links[most_popular_index])