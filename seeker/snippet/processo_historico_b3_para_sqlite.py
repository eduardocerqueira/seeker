#date: 2022-11-02T17:16:01Z
#url: https://api.github.com/gists/f17caa6c66cd2b96c5715edeb6b624a3
#owner: https://api.github.com/users/carlosdelfino

from datetime import datetime
from pytz import timezone

tz = timezone('America/Fortaleza')
data_e_hora_atuais = datetime.now()
data_e_hora_atuais_tz = data_e_hora_atuais.astimezone(tz)

#######################################################

import os
from pathlib import Path
import requests as req

def get_cotacoes(ano, mes=None, dia=None, overwrite=True):
    
    if dia and mes:
        zip_file_name = "COTAHIST_D{}{}{}.ZIP".format(ano,mes,dia);
    elif mes:
        zip_file_name = "COTAHIST_M{}{}.ZIP".format(mes,dia);
    else:
        zip_file_name = "COTAHIST_A{}.ZIP".format(ano);

    dest_path_file = Path("cotacoes/" + zip_file_name)
    if dest_path_file.is_file() and not overwrite:
        print("Arquivo {} já existe, não será baixado!".format(zip_file_name))
        return

    print("Obtendo histórico {}".format(zip_file_name))
    url = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/"+zip_file_name
    headers = { 'accept': '*/*',
      'accept-language': 'en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7,es-MX;q=0.6,es;q=0.5',
      'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
      'x-requested-with': 'XMLHttpRequest',
      'sec-ch-ua': '" Not;A Brand";v="99", "Google Chrome";v="97", "Chromium";v="97"',
      'sec-ch-ua-mobile': '?0',
      'sec-ch-ua-platform': '"macOS"',
      'sec-fetch-dest': 'empty',
      'sec-fetch-mode': 'cors',
      'sec-fetch-site': 'same-origin',
      'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)'}
    get_response = req.get(url,stream=True, headers = headers)
    print('#', end='')
    
    file_name  = "./cotacoes/" + zip_file_name
    os.makedirs("./cotacoes", exist_ok=True)
    
    with open(file_name, 'wb') as f:
        print('#', end='')
        
        for chunk in get_response.iter_content(chunk_size=1024):
            if chunk: 
                print('.', end='')
                f.write(chunk)

        print('#')

#######################################################

from zipfile import ZipFile
import pandas as pd
import os
from pathlib import Path
from sqlite_utils import Database, suggest_column_types

def processa_cotacoes(ano, mes=None, dia=None, overwrite=True):
    
    if dia and mes:
        zip_file_name = "COTAHIST_D{}{}{}.ZIP".format(ano,mes,dia);
    elif mes:
        zip_file_name = "COTAHIST_M{}{}.ZIP".format(mes,dia);
    else:
        zip_file_name = "COTAHIST_A{}.ZIP".format(ano);

    zip_file_name = "cotacoes/" + zip_file_name
    print('#', end='')
    #
    os.makedirs("./cotacoes/database", exist_ok=True)
    db = Database("cotacoes/database/historico_b3.db")
    #
    with ZipFile(zip_file_name, 'r') as zip:
        arq_cotacoes = "COTAHIST_A{}.TXT".format(ano)
        print('#', end='')
        with zip.open(arq_cotacoes) as cotacoes:
            print('#', end='')
            #
            count = 0
            for linha in cotacoes:
                count = count + 1
                dic = {}
                if linha[0:2] == '01':  # linhaistro de dados
                    dic['DATAPlinha'] = datetime.strptime(linha[2:10], '%Y%m%d').astimezone(tz)
                    dic['CODBDI'] = linha[10:12].strip()
                    dic['CODNEG'] = linha[12:24].strip()
                    dic['TPMERC'] = int(linha[24:27])
                    dic['NOMRES'] = linha[27:39].strip()
                    dic['ESPECI'] = linha[39:49].strip()
                    dic['PRAZOT'] = int(linha[49:52]) if linha[49:52].strip() else 0
                    dic['MODREF'] = linha[52:56].strip()
                    dic['PREABE'] = float(linha[56:69])/100
                    dic['PREMAX'] = float(linha[69:82])/100
                    dic['PREMIN'] = float(linha[82:95])/100
                    dic['PREMED'] = float(linha[95:108])/100
                    dic['PREULT'] = float(linha[108:121])/100
                    dic['PREOFC'] = float(linha[121:134])/100
                    dic['PREOFV'] = float(linha[134:147])/100
                    dic['TOTNEG'] = int(linha[147:152])
                    dic['QUATOT'] = int(linha[152:170])
                    dic['VALTOT'] = float(linha[170:188])/100
                    dic['PREEXE'] = float(linha[188:201])/100
                    dic['INDOPC'] = int(linha[201:202])
                    dic['DATVEN'] = datetime.strptime(linha[202:210], '%Y%m%d').astimezone(tz)
                    dic['FATCOT'] = int(linha[210:217])
                    dic['PTOEXE'] = float(linha[217:230])/1000000
                    dic['CODISI'] = linha[230:242].strip()
                    dic['DISMES'] = int(linha[242:245])
                    db['cotacoes'].insert(dic)
                elif linha[0:2] == '00': # registro de metadados
                    print("Arquivo criado em {}".format(datetime.strptime(linha[23:30], '%Y%m%d')))
                elif linha[0:2] == '99': # registro de metadados
                    size = int(linha[31:42])
                    if count != size:
                        raise Exception("Arquivo Invalid, número de linhas diferente: foram processadas {}, mas era esperado {}".format(count, size))
                    #
                #                
                print('.', end='')
                if not count % 40:
                    print(" " + str(count)) 
        zip.close()
    #
    print("# total: {}".format(count))
    return



#######################################################

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    for ano in range(2021,2022):
        #get_cotacoes(ano=ano)
        processa_cotacoes(ano=ano)
    
