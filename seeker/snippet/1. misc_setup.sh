#date: 2021-09-07T17:15:50Z
#url: https://api.github.com/gists/3bfe39589fbe4dafd4f79fb956d59ff3
#owner: https://api.github.com/users/doytsujin

$ cd /opt/nginx/html
$ echo "<cross-domain-policy>
   <allow-access-from domain='*.your_domain.com' />
</cross-domain-policy>" > crossdomain.xml

$ echo "User-Agent: *
Disallow: /" > robots.txt