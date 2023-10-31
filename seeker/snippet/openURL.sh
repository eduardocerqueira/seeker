#date: 2023-10-31T16:44:30Z
#url: https://api.github.com/gists/4ca58b05ada72b3d6b3622c6aa013deb
#owner: https://api.github.com/users/33gl00

###############################
# Open firefox with urls list #
###############################

urls=("https://url1.fr" "https://url2.fr" "https://url3.fr")
for url in "${urls[@]}"; do
    firefox --new-tab "$url"
done