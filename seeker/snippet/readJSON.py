#date: 2022-02-01T17:10:52Z
#url: https://api.github.com/gists/314a19c16e0c727fe001f4927ea63ac9
#owner: https://api.github.com/users/ArieClone

# Written for my Econ 1923 course at Pitt 

# importing the packages first
import json
import os

def readJSONfile(fname):
    
    #first, check if fname is a string
    if type(fname) == str:  # the input is of type string 
        
        #second, check if a file of the name fname exists
        if os.path.exists(fname): # a file of the name strored in fname exists
            
            # we can now open the file
            f = open(fname)    # f is of type _io.TextIOWrapper
            content = f.read() # we want the content of f (the text inside the file f)
            
            #third, verify that the content is valid json
            try:
                json_object = json.loads(content) #try to convert the content of f to a dictionary
                
            except ValueError as e:  # error occured, it wasn't a valid json file
                errMsg = 'The file does not contain valid json'
                return errMsg
            
            # no error occured we can convert to a dictionary
            #JSONdict = json.load(f)
            f.close
            return json_object

            
        else:                     # a file of the name stored in fname does not exist
            errMsg = 'file does not exist'  
            return errMsg
    
    else:                   # the input is not of type string
        errMsg = 'input should be a string'
        return errMsg
    