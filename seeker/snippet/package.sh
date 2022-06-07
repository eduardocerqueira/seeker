#date: 2022-06-07T16:55:22Z
#url: https://api.github.com/gists/902fbaf5cda2bddf6d3bf4c978276f05
#owner: https://api.github.com/users/davidfdr

rm *.tgz
rm *.tar.gz
tar cfz code.tar.gz connection.json ../META-INF
tar cfz ccskuproductstockaccount.tgz metadata.json code.tar.gz