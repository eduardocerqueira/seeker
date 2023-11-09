#date: 2023-11-09T16:47:19Z
#url: https://api.github.com/gists/b613325b7015b5daca4a90c4c6f758bf
#owner: https://api.github.com/users/Luke-in-the-sky

import re
import requests
from bs4 import BeautifulSoup

class ProjectGutenbergAPIError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ProjectGutenbergAPI:
    def __init__(self):
        self.base_url = "https://www.gutenberg.org/ebooks/search/?query="

    def search_books(self, query, top_n=1):
        search_url = self.base_url + query
        response = requests.get(search_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        books = []
        for row in soup.find_all('li', class_='booklink'):
            if top_n == 0:
                break
            link = row.find('a', class_='link')
            book_link = "https://www.gutenberg.org" + link.get('href')
            book_title = row.find('span', class_='title').get_text().strip()
            author = row.find('span', class_='subtitle').get_text().strip()
            books.append({'title': book_title, 'author': author, 'link': book_link})
            top_n -= 1
        return books

    def download_book_content(self, book_link, text_only=True):
        try:
            if text_only:
                book_id_match = re.search(r'gutenberg\.org/ebooks/(\d+)/?$', book_link)
                if book_id_match:
                    book_id = book_id_match.group(1)
                    content_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                    response = requests.get(content_url)
                else:
                    raise ProjectGutenbergAPIError("Failed to extract book_id from book_link")
            else:
                response = requests.get(book_link)

            if response.status_code == 200:
                return response.text
            else:
                raise ProjectGutenbergAPIError("Failed to download book content")
        except requests.exceptions.RequestException as e:
            raise ProjectGutenbergAPIError("Failed to retrieve book content") from e

# Instantiate the API and test the search and download functionality
api = ProjectGutenbergAPI()
try:
    search_results = api.search_books('the odyssey', top_n=1)
    if search_results:
        book_link = search_results[0]['link']
        book_content = api.download_book_content(book_link)
        print(book_content)
    else:
        print("No search results found.")
except ProjectGutenbergAPIError as e:
    print(f"Error: {e}")
