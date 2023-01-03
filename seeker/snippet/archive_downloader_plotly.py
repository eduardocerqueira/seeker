#date: 2023-01-03T17:09:59Z
#url: https://api.github.com/gists/9dc815610bfeef57b79c3f41cf6967fc
#owner: https://api.github.com/users/mariopesch

# -*- coding: utf-8 -*-
"""
openSenseMap Archive Downloader
set Name, BoxID, SensorID and Start Date and Number of Days
"""

import os
import glob
import pandas as pd
import requests
import datetime
import plotly.express as px

#Values to set
start_date = "01-12-2022"
end_date = "02-12-2022"
boxID = "5e98843845f937001cf26c6d"
name = "Nienberge_Garten"
sensorID = "5e98843845f937001cf26c73"

 
# number of days
numberOfDays = 100

#download csv files from archive.opensensemap.org
#save them in the folder "files"
#combine them into one csv file
#set working directory

os.mkdir("files/"+sensorID+"/")
os.chdir("files/"+sensorID+"/")


# initializing date, set start Date
start_date = datetime.datetime.strptime(start_date, "%d-%m-%Y")
end_date = datetime.datetime.strptime(end_date, "%d-%m-%Y")

 
date_generated = pd.date_range(start=start_date, end=end_date )
print(date_generated.strftime("%d-%m-%Y"))

for date in date_generated:
    date = date.strftime("%Y-%m-%d")
    url = "https://archive.opensensemap.org/"+date+"/"+boxID+"-"+name+"/"+sensorID+"-"+date+".csv"
    with requests.get(url, stream=True) as response:
        #response.raise_for_status()
        #print(response.status_code)
        if response.status_code == 200:
            with open(sensorID+"-"+date+".csv", "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                file.flush()
    
print("Done Downloading")

#find all csv files in the folder
#use glob pattern matching -> extension = 'csv'
#save result in list -> all_filenames
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#print(all_filenames)

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
startDate = start_date.strftime("%d-%m-%Y")
endDate = end_date.strftime("%d-%m-%Y")
combined_csv.to_csv(sensorID+"-"+name+"-"+startDate+"-"+endDate+".csv", index=False, encoding='utf-8-sig')

#read csv file
df = pd.read_csv(sensorID+"-"+name+"-"+startDate+"-"+endDate+".csv")

fig = px.scatter(df, x="createdAt", y="value", title='Temperatur in °C')
fig.show()
