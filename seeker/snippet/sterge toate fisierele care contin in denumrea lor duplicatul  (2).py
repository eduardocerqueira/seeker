#date: 2025-02-21T16:45:30Z
#url: https://api.github.com/gists/5c8f881cd82a728ebc30e9caac08d3bd
#owner: https://api.github.com/users/me-suzy

import os

def delete_duplicate_files(folder_path):
    try:
        # Listăm toate fișierele din folder
        files = os.listdir(folder_path)

        # Contoare pentru statistici
        deleted = 0
        errors = 0

        # Căutăm și ștergem fișierele cu (2) în nume
        for file in files:
            if " (2)" in file:
                file_path = os.path.join(folder_path, file)
                try:
                    os.remove(file_path)
                    print(f"Șters: {file}")
                    deleted += 1
                except Exception as e:
                    print(f"Eroare la ștergerea {file}: {str(e)}")
                    errors += 1

        # Afișăm statistici
        print(f"\nRezultat:")
        print(f"- Fișiere șterse: {deleted}")
        print(f"- Erori: {errors}")

    except Exception as e:
        print(f"Eroare la accesarea folderului: {str(e)}")

# Specificăm folderul
folder_path = "g:\\Download"

# Rulăm funcția
delete_duplicate_files(folder_path)