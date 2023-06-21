#date: 2023-06-21T16:35:49Z
#url: https://api.github.com/gists/cbd5cee1e6f3db5356e787467d2e5316
#owner: https://api.github.com/users/KrisAff84

# Imports items to a DynamoDB table from a standard text file with attributes separated by a delimiter
# and items separated by line in the text file

import boto3

def textfile_to_list(txt_file, delimiter):      # defines function textfile_to_list
    file = open(txt_file)                       # opens the file path set to parameter txt_file
    textlist = []                       # declare list
   for line in file:                    # for every line in the file, do the following
        l = line.strip('\n')                # string the '\n' newline character from the end of each line
        textlist.append(l.split(delimiter))     # split line into items between dilimiter parameter
        print(textlist[counter])                # print each line in the file as a list
    return textlist                         # returns list from function

    
def add_items_to_table(table, txt_file, delimiter):       # defines function to add items to dynamoDB table
    textlist = textfile_to_list(txt_file, delimiter)      # we get textlist from the function textfile_to_list
    ddb = boto3.client('dynamodb')                      # sets dbb interface to client
    for entry in textlist:                            # for every entry in textlist do the following
        singer = entry[2]                           # sets 'singer' to 3rd index of each entry
        song = entry[0]                           # sets 'song' to first index of each entry
        key = entry[1]                          # sets 'key' to second index of each entry
        if len(entry) > 3:                      # if 4th index exists, set 'review' to 'Review'
            review = 'Review'
        elif len(entry) <= 3:               # if 4th index does not exist set 'review' to ' '
            review = ' '
        response = ddb.put_item(            # puts items in table
            TableName=table,                # assigns 'TableName' to 'table'
            Item={
                'Singer': {             # assigns 'Singer' to 'singer' with type string
                    'S': singer, 
                },
                'Song': {               # assigns 'Song' to 'song' with type string
                    'S': song,
                },
                'Key': {                # assigns 'Key' to 'key' with type string
                    'S': key,
                },
                'Review': {             # assigns 'Review' to 'review' with type string
                    'S': review,
                },
            },
        )
    return response

def main():
    txt_file = 'Python_boto3/DynamoDB/song_list.txt'        # sets txt_file to path of file to be scanned
    delimiter = ':'                                       # sets delimiter to ':' colon
    table = 'Songs'                                       # sets table to 'Songs'
    add_items_to_table(table, txt_file, delimiter)          # adds items in txt_file to table
    
    
if __name__ == '__main__':
    main()
    