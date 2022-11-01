#date: 2022-11-01T17:16:06Z
#url: https://api.github.com/gists/51481194b2d0d451b0ff04a208b2b284
#owner: https://api.github.com/users/code-and-dogs

url = 'https://www.mobygames.com/browse/games/switch/list-games/'
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')

pageNumbers = int(soup.find('td', class_='mobHeaderPage').text.split(' ')[4])
for bigIteration in range(pageNumbers):
    if bigIteration > 0:
        page = requests.get(nextPage)
        soup = BeautifulSoup(page.text, 'html.parser')
    [...]
    nextPage = soup.find('a', string='Next')['href']   