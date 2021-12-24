#date: 2021-12-24T17:19:52Z
#url: https://api.github.com/gists/e93ac7c14e10f7ae6952f6c039414f2a
#owner: https://api.github.com/users/AngieEspinosa97

import  requests
import  pandas      as pd
from    pprint      import pprint

class EBProperties():

    #-------------------Call API-------------------------#

    url     =   "https://api.stagingeb.com/v1/properties"
    apikey  =   "l7u502p8v46ba3ppgvj5y2aad50lb9"
    headers =   {'Content-Type': 'application/json', 'X-Authorization': apikey}

    #------------------Build GET request------------------#

    def set_url(url, headers, x):
        response= requests.get(url + f'?page={x}&limit=50', headers=headers)
        return response.json()
    
    #-----------------Inquire total pages-----------------#

    def get_pages(result):
        return result["pagination"]["page"]

    #----------------Pick titles out and Wrap them---------------#

    def get_all_properties(result):
        propertiesList=[]
        for item in result["content"]:
            char =  {
                "Titulos_de_Propiedades": item["title"]
            }
            for title in char:
                #print(char[title]) => verify wheter console is printing single tittles 
                propertiesList.append(char[title])
        return propertiesList
    
    #-----------------List Total titles properties-----------------------#

    TitlesPropertiesList = []
    data = set_url(url, headers, 1)
    for x in range(1,get_pages(data)+16):
        #print(x)
        TitlesPropertiesList.extend(get_all_properties(set_url(url, headers, x)))
        
    #-----------alphabetically print -----------------#
    alpha = sorted(TitlesPropertiesList)
    #pprint(alpha)
    
    print(len(TitlesPropertiesList)) #=> to verify the pages and the 766 titles in total

    #--------------Store data in a file----------------#

    df= pd.DataFrame(TitlesPropertiesList)
    df.to_html("propertiesList.html", index= False)
    df.to_csv("propertiesList.csv", index= False)