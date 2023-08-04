#date: 2023-08-04T16:58:41Z
#url: https://api.github.com/gists/f17a371acecfcadedca4305a1aa79ff9
#owner: https://api.github.com/users/Ancreem

import json
import csv 

def cargar(file_name):
    try:
        with open(file_name, "r") as file:
            data= json.load(file)
            return data
    except FileNotFoundError:
        return {}
    except Exception:
        return{}
    
def guardar(file_name, datos):
    try:
        with open(file_name, "w") as file:
            json.dump(datos, file)
            return True
    except FileNotFoundError:
        return False
    except Exception:
        return False
    
def guardar_registro(file_name,datos):
    try:
        with open(file_name, "a", newline="", encoding="UTF-8") as file:
            writer=csv.writer(file)
            writer.writerow(datos)
            return True
    except FileNotFoundError:
        return False
    except Exception:
        return False