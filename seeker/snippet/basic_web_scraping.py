#date: 2022-12-13T16:44:44Z
#url: https://api.github.com/gists/fe640224c94e1a743c7287427707d9fa
#owner: https://api.github.com/users/gigafide

#IMPORT LIBRARIES
from bs4 import BeautifulSoup
import requests

#REQUEST WEBPAGE AND STORE IT AS A VARIABLE
page_to_scrape = requests.get("http://quotes.toscrape.com")

#USE BEAUTIFULSOUP TO PARSE THE HTML AND STORE IT AS A VARIABLE
soup = BeautifulSoup(page_to_scrape.text, 'html.parser')

#FIND ALL THE ITEMS IN THE PAGE WITH A CLASS ATTRIBUTE OF 'TEXT'
#AND STORE THE LIST AS A VARIABLE
quotes = soup.findAll('span', attrs={'class':'text'})

#FIND ALL THE ITEMS IN THE PAGE WITH A CLASS ATTRIBUTE OF 'AUTHOR'
#AND STORE THE LIST AS A VARIABLE
authors = soup.findAll('small', attrs={"class":"author"})

#LOOP THROUGH BOTH LISTS USING THE 'ZIP' FUNCTION
#AND PRINT AND FORMAT THE RESULTS
for quote, author in zip(quotes, authors):
    print(quote.text + "-" + author.text)