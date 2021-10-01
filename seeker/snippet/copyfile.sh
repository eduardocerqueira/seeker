#date: 2021-10-01T16:57:17Z
#url: https://api.github.com/gists/b3196c9476d5259f78116b236af6aac2
#owner: https://api.github.com/users/fraigo

FILEID=xxxx
ACCESS_TOKEN=yyyyy
curl --request POST \
  "https://www.googleapis.com/drive/v3/files/$FILEID/copy" \
  --header "Authorization: Bearer $ACCESS_TOKEN" \
  --header 'Accept: application/json' \
  --header 'Content-Type: application/json' \
  --data '{"mimeType":"application/x-msmetafile","name":"copiedfile.emf"}' \
  --compressed