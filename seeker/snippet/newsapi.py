#date: 2021-12-13T16:57:48Z
#url: https://api.github.com/gists/61bddb5bd6a511ccee6be79d956c93f7
#owner: https://api.github.com/users/CrimsonScythe

from flask import Flask, request
from calc import getCovidData, genReport
from celery import chord
from default_settings import *

app = Flask(__name__)

@app.route('/api/send', methods=['GET', 'POST'])
def send():

    countries = request.get_json()["countries"]

    ''' Fast version '''
    tasks = [getCovidData.s(country) for country in countries]
    res = chord(tasks)(genReport.s())
    df = res.get()

    '''Slow version'''
    # reslst=[]
    # for ye in countries:
    #     res = requests.request("GET", "https://newsapi.org/v2/top-headlines",
    #     params={"country":ye, "apiKey":"28654dee489f4d1aac5b6d59f69d62bf"})
    #     reslst.append(res.json())

    # df = genReport.delay(reslst).get()

    return df
   
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
