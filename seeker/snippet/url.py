#date: 2022-03-07T17:00:41Z
#url: https://api.github.com/gists/7725eb96744f28bed26e3f0044309d2d
#owner: https://api.github.com/users/irenegyebi

# pip install pyshorteners
import pyshorteners
url=input("Paste url here")
shortener=pyshorteners.Shortener()
output=shortener.tinyurl.short(url)
print(output)