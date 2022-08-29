#date: 2022-08-29T16:52:25Z
#url: https://api.github.com/gists/06889dbe0b84645d657609ab77a6e7bc
#owner: https://api.github.com/users/BaoNgocPham

import requests
from bs4 import BeautifulSoup
import pprint # make it print better

res = requests.get("https://news.ycombinator.com/news") #this request like borowser Chrome...

soup = BeautifulSoup(res.text, 'html.parser')

# print(soup.body) #extract only body content
# print(soup.find_all("div"))
# print(soup.find("a")) # find the first item

link = soup.select(".titlelink") #title of the link we need
subtext = soup.select(".subtext") #we need the dot right infront of subtext to extract it out from website

def sorted_list_by_vote(hnlist): #sort the vote from highest to lowest
    return sorted(hnlist, key = lambda k:k["vote"], reverse = True) #sort base on vote in dict

def create_custom_hn(links, subtext):
    hn = []
    for idx, item in enumerate(links):  #extract index
        title = item.getText()
        href = item.get("href", None) #grab the hyperlink
        vote = subtext[idx].select(".score")
        if len(vote):
            points = int(vote[0].getText().replace(" points", " ")) #get the point number from Vote (if you print vote, you will see why)
            if points > 99:
                hn.append({"title": title, "links": href, "vote" : points})  # add into list
    return sorted_list_by_vote(hn)

hn1 = create_custom_hn(link,subtext)

pprint.pprint(hn1)

