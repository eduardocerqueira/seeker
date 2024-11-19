#date: 2024-11-19T16:47:38Z
#url: https://api.github.com/gists/215ad8e88e3ef46854af066c1ff0da54
#owner: https://api.github.com/users/MrNtex

import requests
from bs4 import BeautifulSoup
from fpdf import FPDF

def scrap_pokemon_list():
  url='https://pokelife.pl/pokedex/index.php?title=Lista_Pokemon%C3%B3w_Kanto'
  response = requests.get(url)
  
  soup = BeautifulSoup(response.content, 'html.parser')
  pokemons_list = []

  table= soup.find('table', class_='Tabela1')
  rows = table.find_all('tr')
  
  for row in rows[1:]:
    columns = row.find_all('td')
    pokemon = (columns[2].text[:-1], f"No.{columns[0].text[:-1]}")
    pokemons_list.append(pokemon)

  return pokemons_list

def get_pokemon_info(pokemon_name):
  pokemon_name = pokemon_name.replace('\'','')
  pokemon_name = pokemon_name.replace('. ','-')
  pokemon_name = pokemon_name.replace('♀','-f')
  pokemon_name = pokemon_name.replace('♂','-m')
  url = f'https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}'
  response = requests.get(url)
  return response.json()

  
pokemon_list = scrap_pokemon_list()

for pokemon in pokemon_list:
  try:
    pokemon_info = get_pokemon_info(pokemon[0])
  except:
    print("Blad: " + pokemon[0])
  else:
    print(pokemon_info['name'])

print(pokemon_list)

