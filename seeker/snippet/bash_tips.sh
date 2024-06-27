#date: 2024-06-27T16:46:42Z
#url: https://api.github.com/gists/7e0632efea41e9bbf7a6e56892a81c9b
#owner: https://api.github.com/users/klapauciuz

# get first column(from file with ':' delimeter) and write into output.csv
cut -f1 -d':' input.csv >> output.csv
# get from first up to five column from dataset.csv
cut -f1-5 -d',' dataset.csv

# regex for json-type html parsing
\:(?:[^\:]*)@\C+

# get first column from output.csv
awk -F, '!seen[$1]++' output.csv >> uuu.csv

# get lines starts from number 2261627 to 2402947p from output.csv file
sed -n 2261627,2402947p output.csv

# download file from ssh server
scp root@127.0.0.1:/root/project/output.csv /Users/username/Desktop/project/output.csv

# sort and filter only unique lines from ids.txt
sort -u ids.txt >> uniq_ids.txt
