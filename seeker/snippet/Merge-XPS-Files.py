#date: 2024-08-22T16:52:00Z
#url: https://api.github.com/gists/967d657bce67703dc850da1541272d7b
#owner: https://api.github.com/users/aspose-com-gists

from aspose.page.xps import * 

# Define the working directory.
data_dir =  "./files"
# Initialize the XPS output stream
with open(data_dir + "mergedXPSfiles.xps", "wb") as out_stream:
    # Load XPS document from the file by instantiating an instance of the XpsDocument class. 
    document = XpsDocument(data_dir + "input.xps", XpsLoadOptions())
    # Define an array of XPS files which will be merged with the first one.
    files_to_merge = [ data_dir + "Demo.xps", data_dir + "sample.xps" ]
    # Invoke the merge method to merge the XPS files. 
    document.merge(files_to_merge, out_stream)