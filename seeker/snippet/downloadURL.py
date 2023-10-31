#date: 2023-10-31T16:46:47Z
#url: https://api.github.com/gists/14cbe52e17094e3946621257dee9d2df
#owner: https://api.github.com/users/33gl00

############################
# Download files from urls #
############################

import os
import requests


import argparse
parser = argparse.ArgumentParser(
                    prog = 'download Url List',
                    description = 'download source from url file list',
                    epilog = 'python3 downloadUrlist.py --path urList.txt ')

parser.add_argument('-p', '--path', type=str, required=True, help="")
input = parser.parse_args()
path = input.path

def download(filePath):
    # Ouvre le fichier contenant les URLs
    with open(filePath, 'r') as file:
        urls = file.readlines()

    # Initialise un compteur d'ID
    id_counter = 1

    # Boucle à travers chaque URL
    for url in urls:
        url = url.strip()  # Supprime les espaces et les retours chariot

        # Obtiens le nom du fichier à partir de l'URL
        file_name = url.split("/")[-1]
        # Ajoute l'ID au nom du fichier pour éviter les conflits de doublons
        file_name = f"{id_counter}_{file_name}"
        id_counter += 1

        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, 'wb') as output_file:
                output_file.write(response.content)
            print(f"Le fichier {file_name} a été téléchargé avec succès.")
        else:
            print(f"Échec du téléchargement du fichier depuis l'URL : {url}")

download(path)
exit()
####### eegloo