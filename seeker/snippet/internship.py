#date: 2021-12-28T16:48:35Z
#url: https://api.github.com/gists/7d5b2e3a6e6e45f399e64606e5d21489
#owner: https://api.github.com/users/nicmedina

"""
Created on Mon Dec 27 17:20:35 2021

@author: Nicolas
"""

import requests   #import libraries
import json

class getProperties:  #create the simple class
    def __init__(self,url,headers,args):  #init object with parameters 
        self.url = url
        self.headers = headers
        self.args = args
        
    def printProperties(self):  #method to read and print the titles
        response = requests.get(self.url, headers=self.headers, params=self.args) #get the response from URL
        if response.status_code == 200:   #check access granted on website
            response_json = json.loads(response.text)   #load json text 
            content = response_json['content']  #obtain dictionary content
            pagination = response_json['pagination']  #obtain dictionary pagination
            limit = pagination['limit'] #get the limit of properties per page
            i = 0
            titles = ['none']*20    #init list titles to compare afterwards
            while i < limit:      #cycle the content dictionary searching for the titles
                title = content[i]
                titleA = title['title']
                titles[i] = titleA
                print(titleA)   #prints the titles as requested 
                i += 1
            return titles  #return the list of the titles 