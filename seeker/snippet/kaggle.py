#date: 2022-10-14T17:22:38Z
#url: https://api.github.com/gists/184e1270e3f1bac3f9457e016d62bac3
#owner: https://api.github.com/users/florent-brosse

if [ -n "$KAGGLE_USER" ]; then
 mkdir -p /root/.kaggle
 echo '{"username": ${KAGGLE_USER},"key": ${KAGGLE_APIKEY}}' > /root/.kaggle/kaggle.json
 chmod 600 /root/.kaggle/kaggle.json
 pip install kaggle
fi