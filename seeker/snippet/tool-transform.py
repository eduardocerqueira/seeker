#date: 2022-02-10T17:06:47Z
#url: https://api.github.com/gists/08d09c9f6a124d6f9c23aea8e07b5dbb
#owner: https://api.github.com/users/goish135

# V Read every line in application.config
# V split each line using equal operator 

# ---
# write to excel: 
# Put key value to excel specified field 
# & default value already wriien in excel 
# ---
# [Note]
# Check it ... 
# Diff env diff value but same key 
# File ref in para 
# Erb context & mapping (file location)
# # dic name: dic
# dic = {}

with open('/Users/supinyu/Desktop/sample.config') as f:
    for line in f:
        print(line)
        arr=line.rsplit("=")
        print("key:"+arr[0]+"\t"+"val:"+arr[1]) # to be remove head tail space or not ?
        dic[arr[0]] = arr[1].strip('\n')
print(dic)