#date: 2024-12-19T17:04:04Z
#url: https://api.github.com/gists/f02d2f6af80543b4bcce3c440500bf11
#owner: https://api.github.com/users/me-suzy

import os
import re

directory = r'e:\Carte\BB\17 - Site Leadership\Principal 2022\en'

def process_files():
    print(f"Începem procesarea fișierelor din {directory}")

    for filename in os.listdir(directory):
        if not filename.endswith('.html') or not filename.startswith('webinar'):
            continue

        file_path = os.path.join(directory, filename)
        print(f"\nProcesez fișierul webinar: {filename}")

        try:
            # Citește conținutul fișierului
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Primul regex: găsește secțiunea COPERTA
            coperta_sections = re.finditer(r'<!-- COPERTA START -->[\s\S]*?<!-- COPERTA FINAL -->', content)

            new_content = content
            for match in coperta_sections:
                coperta_section = match.group(0)
                print(f"Găsit secțiune COPERTA")

                # Al doilea regex: înlocuiește src="index_files cu src="https://...
                new_section = re.sub(
                    r'(<img src=")index_files/',
                    r'\1https://neculaifantanaru.com/en/index_files/',
                    coperta_section
                )

                if new_section != coperta_section:
                    print(f"Înlocuire făcută în secțiunea COPERTA")
                    new_content = new_content.replace(coperta_section, new_section)

            if new_content != content:
                print(f"Salvez modificările în {filename}")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Modificări salvate cu succes în {filename}")
            else:
                print(f"Nu sunt necesare modificări în {filename}")

        except Exception as e:
            print(f"Eroare la procesarea {filename}: {str(e)}")

if __name__ == "__main__":
    process_files()