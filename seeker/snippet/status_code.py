#date: 2021-12-03T17:05:20Z
#url: https://api.github.com/gists/9d00ee85b3871f3c1249d3fde52f822a
#owner: https://api.github.com/users/dgtlctzn

...
async with session.get(url) as response:
    if response.status == 503:
	# do some work