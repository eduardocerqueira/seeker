#date: 2023-01-10T16:53:14Z
#url: https://api.github.com/gists/d9b1828502bd358caed2bafa043c0a4a
#owner: https://api.github.com/users/danizavtz

import requests, pandas as pd, json
from requests import RequestException

def getAPI():
    url = "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoMoedaPeriodo(moeda=@moeda,dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)?@moeda='USD'&@dataInicial='01-01-2022'&@dataFinalCotacao='01-05-2023'&$top=100000000&$format=json&$select=paridadeCompra,paridadeVenda,cotacaoCompra,cotacaoVenda,dataHoraCotacao,tipoBoletim"

    payload={}
    headers = {}
    try:
        response = requests.request("GET", url, headers=headers, data=payload)
        apenasAtributoValue = response.json()['value'] #esta linha faz o que deseja
        df = pd.DataFrame.from_dict(apenasAtributoValue)
        print(df)
                
    except RequestException as error:
                print(error)
    
getAPI()