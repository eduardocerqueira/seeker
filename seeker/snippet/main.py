#date: 2024-06-04T16:59:43Z
#url: https://api.github.com/gists/4921b20351f92b40c44097bff3f9b8c0
#owner: https://api.github.com/users/tails1405

import cpuinfo
import os
import requests
person = []
choix = "0"
print("                  __                                      __                   .__           ")
print("  ____  ___  ____/  |_ _______   ____    _____    ____  _/  |_   ____    ____  |  |    ______")
print("_/ __ \ \  \/  /\   __\\_  __ \_/ __ \  /     \ _/ __ \ \   __\ /  _ \  /  _ \ |  |   /  ___/")
print("\  ___/  >    <  |  |   |  | \/\  ___/ |  Y Y  \\  ___/  |  |  (  <_> )(  <_> )|  |__ \___ \ ")
print(" \___  >/__/\_ \ |__|   |__|    \___  >|__|_|  / \___  > |__|   \____/  \____/ |____//____  >")
print("     \/       \/                    \/       \/      \/                                   \/ ")
while choix != "4" or choix ==1 or choix ==2 or choix ==3:

    # Gestion d'une liste du personnel
    print("**********************************")
    print("1)info cpu")
    print("2)ip position")
    print("3)Effacer une personne de la liste")
    print("4)Fin")
    print("")
    choix = (input("Entrez votre choix : "))
    if choix == "1":
        #recherche et variable
        nom_coeur = cpuinfo.get_cpu_info()["count"]
        marque_coeur = cpuinfo.get_cpu_info()["brand_raw"]
        frequence_processure = cpuinfo.get_cpu_info()["hz_actual_friendly"]
        #titre
        print("_________  __________  ____ ___ ")
        print("\_   ___ \ \______   \|    |   \ ")
        print("/    \  \/  |     ___/|    |   /")
        print("\     \____ |    |    |    |  / ")
        print(" \______  / |____|    |______/  ")
        print("        \/                      ")
        #affichage resultat
        print(f"processeur : {marque_coeur}")
        print(f"le processeur possède {nom_coeur} coeur(s)")
        print(f"frequence actuelle du processeur : {frequence_processure}")
        os.system("pause")
    elif choix == "2":
        print("                    .__  __  .__               ")
        print("______   ____  _____|__|/  |_|__| ____   ____  ")
        print("\____ \ /  _ \/  ___/  \   __\  |/  _ \ /    \ ")
        print("|  |_> >  <_> )___ \|  ||  | |  (  <_> )   |  \ ")
        print("|   __/ \____/____  >__||__| |__|\____/|___|  /")
        print("|__|              \/                        \/ ")
        ip = input("votre ip: ")

        def get_location():
                ip_address = ip
                response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
                location_data = {
                "ip": ip_address,
                'city': response.get('city'),
                'region': response.get('region'),
                'country': response.get('country_name')
                }
                return location_data
    
        get_location()

        print(get_location())
        os.system("pause")
    elif choix == "3":
        person.remove(input("Entrer le nom prénom de la personne à effacer : "))
        print("")