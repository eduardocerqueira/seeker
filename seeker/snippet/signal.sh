#date: 2022-02-10T16:45:04Z
#url: https://api.github.com/gists/39bb16966fbbebe751c45e60d6cbd1c2
#owner: https://api.github.com/users/junefish

curl https://updates.signal.org/desktop/apt/dists/xenial/main/binary-amd64/Packages.gz | zcat | grep Filename | sed 's_Filename: _https://updates.signal.org/desktop/apt/_'