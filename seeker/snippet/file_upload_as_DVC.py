#date: 2022-10-20T17:15:18Z
#url: https://api.github.com/gists/92335af386834c2b9df35adf8e907f82
#owner: https://api.github.com/users/keitazoumana

from dagshub.upload import Repo 
import os

# Instanciate your Repository
my_repo = "**********"=USERNAME, password=PASSWORD)

# Helper function to upload all the files from a folder
def upload_files(folder_name, commit_message):

    dvc_folder = my_repo.directory(folder_name)

    for file in os.listdir(folder_name):

        dvc_folder.add(folder_name+'/'+file) 
    
    # Run the final commit
    dvc_folder.commit(commit_message, versioning="dvc")

"""
Upload my folder
"""
upload_files(data, commit_message="Upload of the data to DVC")oad of the data to DVC")