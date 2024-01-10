#date: 2024-01-10T17:00:20Z
#url: https://api.github.com/gists/6d4afe8c859f9438e0ac90aa77588e57
#owner: https://api.github.com/users/KiefBC

from bs4 import BeautifulSoup
import os
import requests
import string


def request_page(num_page, url):
    my_params = {'sort': 'PubDate', 'year': '2020', 'page': str(num_page)}
    r = requests.get(url, params=my_params)
    return r


def clean_article_title(title):
    clean_title = title.strip().replace(' ', '_')
    for character in clean_title:
        if character not in (string.ascii_letters + string.digits + '_' + '’'):
            clean_title = clean_title.replace(character, '')
    clean_title = clean_title.replace('__', '_')
    return clean_title + '.txt'


def extract_articles(soup, topic):
    articles_url = []
    all_articles = soup.find_all('article')
    for article in all_articles:
        span_tag = article.find('span', {'class': 'c-meta__type'})
        if span_tag.text == topic:
            a_tag = article.find('a', {'data-track-action': 'view article'})
            link = a_tag.get('href')
            articles_url.append('https://www.nature.com' + link)
    return articles_url


def save_article(article_url, dir):
    r = requests.get(article_url)
    soup = BeautifulSoup(r.content, 'html.parser')
    title = soup.find('h1').text
    clean_title = clean_article_title(title)
    body = soup.find('div', {'class': 'c-article-body'}).text.strip()
    with open(os.path.join(dir, clean_title), 'wb') as f:
        f.write(bytes(body, encoding='utf-8'))
    print('article', clean_title, 'saved!')
    return clean_title


def save_page(r, topic, dir):
    soup = BeautifulSoup(r.content, 'html.parser')
    articles = extract_articles(soup, topic)
    names = []
    for art in articles:
        names.append(save_article(art, dir))
    print(names)


def process_page(num_page, topic, url):
    r = request_page(num_page, url)
    name_dir = 'Page_' + str(num_page)
    os.mkdir(name_dir)
    if r:
        save_page(r, topic, name_dir)
    else:
        print(f"Page {num_page} unavailable")


def save_articles():
    print(os.getcwd())
    total_pages = int(input())
    topic = input()
    url_nature = 'https://www.nature.com/nature/articles'
    for num_page in range(total_pages):
        process_page(num_page + 1, topic, url_nature)

save_articles()