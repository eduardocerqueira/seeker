#date: 2022-06-03T17:14:36Z
#url: https://api.github.com/gists/fc11e41fc2fc54aa90bf5d722fb4196b
#owner: https://api.github.com/users/wwwxkz

# This code is not intended for production, used as a shortcute and tool for minor projects
# Pay attention to the results
# Do not rely your job or company in this code

import requests

# Example data
# Using -> SELECT DISTINCT city FROM `random_table`;
municipios = [
'Recife',
'Bragança Paulista',
'Niterói'
]


# Get all cities and states and save in file for easier and faster access 
# First run is slower 
option = input("1: Get data or 2: Test data -> ")
if(option == 1):
  f = open("ibge-municipios.txt", "w")
  f.close()
  f = open("ibge-municipios.txt", "a")
  url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/"
  data_estados = requests.get(url)
  for estado in data_estados.json():
      url_municipios = url + estado['sigla'] + '/municipios/'
      data_municipios = requests.get(url_municipios)
      for municipio in data_municipios.json():
          f.writelines("{}\n".format(municipio['nome']))    
  f.close()

# Uses file and test your locations, returning the invalid ones
if(option == 2):
  fileObj = open('ibge-municipios.txt', "r")
  words = fileObj.read().splitlines()
  fileObj.close()
  municipio_safe = []
  for municipio in municipios:
      for word in words: 
          if (municipio == word):
              municipio_safe.append(municipio)
  for i in municipio_safe:
       if i in municipios:
          municipios.remove(i)
  print(*municipios, sep='\n')


