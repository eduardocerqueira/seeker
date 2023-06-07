#date: 2023-06-07T16:58:24Z
#url: https://api.github.com/gists/c58ce76f5e67ab42b51156b104ad1206
#owner: https://api.github.com/users/rfeers

pathname = "Data/my-take-out/Location History/Semantic Location History/"

def load_and_process_json_files(pathname):
    data = [] #A void list is defined
    all_folders = [x[0] for x in os.walk(pathname)] #Get all the folders contained within the pathname
    for folder in all_folders: #Iterate over every folders
        for filename in os.listdir(folder): #Iterate over all files
            if filename.endswith('.json'):
                with open(os.path.join(folder, filename), 'r') as file:
                    json_data = json.load(file)    #The json data is extracted
                    data.append(json_data)         #The void list is updated with the new data
    return data

data = load_and_process_json_files(pathname)