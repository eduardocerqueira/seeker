#date: 2025-11-27T17:08:52Z
#url: https://api.github.com/gists/55c84c59f001b7abcc906dc937d18898
#owner: https://api.github.com/users/Pallandos

#!/bin/bash

# --- CONFIGURATION ---
# Définir le seuil en nanosecondes.
# Exemple : 5000000 ns = 5 millisecondes
SEUIL="${$1:-5000000}" 

# Initialisation du temps précédent avant d'entrer dans la boucle
# %s = secondes depuis l'époque, %N = nanosecondes
prev_time=$(date +%s%N)

echo "Démarrage de la boucle de surveillance (Ctrl+C pour arrêter)..."
echo "Seuil d'alerte : $SEUIL nanosecondes"

# --- BOUCLE INFINIE ---
while true; do
    # 1. Mesure du temps actuel
    current_time=$(date +%s%N)

    # 2. Calcul de la différence (Delta)
    # $(( ... )) permet de faire des calculs arithmétiques entiers
    delta=$((current_time - prev_time))

    # 3. Comparaison avec le seuil
    if (( delta > SEUIL )); then
        # Optionnel : Conversion en millisecondes pour la lisibilité
        ms=$((delta / 1000000))
        echo "ALERTE : Delta de ${ms} ms ($delta ns) détecté."
    fi

    # Mise à jour du temps précédent pour la prochaine itération
    prev_time=$current_time

    # (Optionnel) Décommente la ligne ci-dessous pour simuler une charge et tester le seuil
    # sleep 0.01 
done