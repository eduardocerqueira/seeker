#date: 2021-12-13T16:57:09Z
#url: https://api.github.com/gists/aadb7a161b481e0dd53b864d44a07bb9
#owner: https://api.github.com/users/CrimsonScythe

from celery import Celery
import requests
import pandas as pd

celery = Celery(
        'calc',
        backend='redis://localhost',
        broker='pyamqp://guest@localhost//'
    )

@celery.task()
def getCovidData(country):
    
    res = requests.request("GET", "https://newsapi.org/v2/top-headlines",
        params={"country":country, "apiKey":"28654dee489f4d1aac5b6d59f69d62bf"})

    return res.json()

@celery.task()
def genReport(results):
    print(results[0])

    dfs=[]

    for result in results:
        df=pd.DataFrame(result)
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0)
    df.index=list(range(df.shape[0]))
        
    return df.to_json()
