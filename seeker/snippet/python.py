//date: 2021-09-01T01:50:09Z
//url: https://api.github.com/gists/8d5cccec38c5c95e96eb447b28b5391f
//owner: https://api.github.com/users/mrclfd

import requests

headers = {
    'User-Agent': 'Chrome/62.0 (BSD x86_64; rv:71.0) Gecko/20100101 Firefox/71.0',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Requested-With': 'XMLHttpRequest',
    'Origin': 'https://s.id',
    'Connection': 'keep-alive',
    'Referer': 'https://s.id/',
}

data = {
  'url': 'https://ne.com'
}

response = requests.post('https://s.id/api/public/link/shorten', headers=headers, data=data)
