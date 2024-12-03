#date: 2024-12-03T16:54:29Z
#url: https://api.github.com/gists/e43680d58c2c5d032ad46b41f2d9ee53
#owner: https://api.github.com/users/KeichiTS

## Read *.csv with pandas ##

#import pandas library 
import pandas as pd 

#By default, sep = ',' and header = 0 
df = pd.read_csv('archive.csv', sep = ';', header = 1)