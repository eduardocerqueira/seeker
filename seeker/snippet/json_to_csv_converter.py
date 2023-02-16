#date: 2023-02-16T16:36:37Z
#url: https://api.github.com/gists/8dd1629f94f7c8fc88bb6d612c1e07c0
#owner: https://api.github.com/users/tenapato

#   
#   Script that parses a json array to a csv file
#   Author: Patricio Tena
#   Date: 2/16/2023

import csv
import glob
import os
import json

# define the input directory path
input_dir = '/path/to/directory/'

# get a list of all JSON files in the input directory
json_files = glob.glob(os.path.join(input_dir, '*.json'))

# print a numbered list of the JSON files in the input directory
print('Select file to convert:')
for i, json_file in enumerate(json_files):
    print(f"{i+1}. {os.path.basename(json_file)}")

# prompt the user to select a file by index
selected_index = int(input('Enter the number of the file to convert: ')) - 1

# load the selected JSON file and extract the headers
selected_file = json_files[selected_index]
with open(selected_file, mode='r') as json_file:
    json_array = json.load(json_file)
    headers = list(json_array[0].keys())

# convert the JSON data to CSV
output_file = os.path.splitext(selected_file)[0] + '.csv'
with open(output_file, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=headers)
    writer.writeheader()
    writer.writerows(json_array)