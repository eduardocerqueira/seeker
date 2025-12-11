#date: 2025-12-11T16:48:27Z
#url: https://api.github.com/gists/9652aa5a9859b7efeeb81c0233f15b48
#owner: https://api.github.com/users/dewomser


import subprocess
from bs4 import BeautifulSoup
import json
import re

def fetch_html_with_curl(url):
    """
    Ruft den HTML-Inhalt von einer gegebenen URL mit curl ab, um Blockaden zu umgehen.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        result = subprocess.run(
            ['curl', '-A', headers['User-Agent'], '-L', url],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Fehler beim Abrufen von {url} mit curl: {e}")
        return None

def parse_www_stadtrat(html_content):
    """
    Parst die HTML-Inhalte der "Worms will weiter" Stadtrat-Seite.
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    politicians = {}

    # Die Seite ist mit Elementor gebaut, die Struktur ist sehr div-lastig.
    # Wir suchen nach den Containern, die eine Person (Bild + Text) enthalten.
    # Die Struktur scheint ein Container für das Bild und ein Container für den Text zu sein.
    # Wir zielen auf den Text-Container, da dieser die Informationen enthält.
    
    # Es gibt zwei Hauptabschnitte: "Ehrenamtlicher Beigeordneter" und "Die Fraktion"
    # sowie einen Grid-Abschnitt für "Gremien und Ausschüsse"
    
    # Regex, um "(stv.)" und ähnliche Suffixe zu extrahieren
    committee_regex = re.compile(r'^(.*?)(?:\s*\((stv\.|stellvertretendes Mitglied)\))?$')
    
    # Funktion, um die Funktionen aus einem Textblock zu extrahieren
    def extract_functions_from_block(text_container, name):
        functions = []
        # Alle <p> Tags im Container extrahieren
        paragraphs = text_container.find_all('p')
        for p in paragraphs:
            # Den <br> getrennten Text verarbeiten
            for br in p.find_all('br'):
                br.replace_with('|||')
            
            lines = p.get_text().split('|||')
            for line in lines:
                line = line.strip()
                if not line or name in line:
                    continue

                # Spezifische einleitende Sätze überspringen
                if line.startswith('ist von Beruf') or \
                   line.startswith('ist der Fraktionsvorsitzende') or \
                   line.startswith('ist der stellvertretende Fraktionsvorsitzende') or \
                   line.startswith('ist seit') or \
                   line.startswith('vertritt unseren Verein'):
                    continue

                # Rollen extrahieren
                if 'Fraktionsvorsitzende' in line:
                    functions.append('Fraktionsvorsitzender')
                elif 'stellvertretende Fraktionsvorsitzende' in line:
                    functions.append('stellvertretender Fraktionsvorsitzender')
                elif 'ehrenamtlicher Beigeordneter' in line:
                    functions.append('Ehrenamtlicher Beigeordneter')
                
                # Gremien extrahieren
                match = committee_regex.match(line)
                if match:
                    committee_name = match.group(1).strip().rstrip(',')
                    is_deputy = match.group(2)
                    
                    if committee_name:
                        full_function = committee_name
                        if is_deputy:
                            full_function += " (stv.)"
                        
                        # Nur hinzufügen, wenn es nicht bereits eine zu allgemeine Funktion ist
                        if full_function.lower() not in [f.lower() for f in functions]:
                           functions.append(full_function)

        # Name und Kontaktzeilen entfernen
        functions = [f for f in functions if name not in f and 'Kontakt:' not in f]
        return list(dict.fromkeys(functions)) # Duplikate entfernen und Reihenfolge beibehalten

    # Finde die Hauptsektionen mit den Personen
    # Dies ist ein Container, der Personen-Widgets enthält.
    # Jeder Personen-Block hat ein Bild und einen Texteditor-Widget.
    person_containers = soup.select('.elementor-widget-text-editor')
    
    for container in person_containers:
        # Den Namen der Person finden (normalerweise im ersten <strong>-Tag)
        name_tag = container.find('strong')
        if name_tag:
            name = name_tag.get_text(strip=True)
            
            # Sicherstellen, dass es sich um einen relevanten Block handelt
            if len(name.split()) > 3 or len(name) < 3: # Heuristik: Namen sind meist 2-3 Wörter lang
                continue

            # Funktionen extrahieren
            functions = extract_functions_from_block(container, name)
            
            if name in politicians:
                existing_functions = set(politicians[name]["functions"])
                new_functions = set(functions)
                politicians[name]["functions"] = sorted(list(existing_functions.union(new_functions)))
            else:
                 politicians[name] = {"name": name, "functions": sorted(list(set(functions)))}

    # Zweiter Durchlauf für die Grid-Ansicht am Ende der Seite
    grid_containers = soup.select('.elementor-element-eaac70b .e-con-full')
    for container in grid_containers:
        name_tag = container.select_one('.elementor-widget-text-editor p:first-child')
        if name_tag:
            name = name_tag.get_text(strip=True)
            if not name or len(name.split()) > 3:
                continue

            functions_tags = container.select('.elementor-widget-text-editor p:not(:first-child)')
            functions = []
            for f_tag in functions_tags:
                for br in f_tag.find_all('br'):
                    br.replace_with('|||')
                lines = f_tag.get_text().split('|||')
                for line in lines:
                    cleaned_line = line.strip().rstrip(',')
                    if cleaned_line:
                        functions.append(cleaned_line)
            
            if name in politicians:
                 politicians[name]["functions"].extend(functions)
                 politicians[name]["functions"] = sorted(list(set(politicians[name]["functions"])))
            else:
                 politicians[name] = {"name": name, "functions": sorted(list(set(functions)))}


    return sorted(list(politicians.values()), key=lambda x: x["name"])


def main():
    """
    Hauptfunktion des Skripts.
    """
    stadtrat_url = "https://wormswillweiter.de/stadtrat"
    output_filename = "www_politiker_worms.json"

    print(f"Rufe HTML von {stadtrat_url} ab...")
    html_content = fetch_html_with_curl(stadtrat_url)

    if not html_content:
        print("Konnte die Webseite nicht abrufen. Das Skript wird beendet.")
        return

    print("Parse Stadtrats-Daten...")
    stadtrat_data = parse_www_stadtrat(html_content)

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(stadtrat_data, f, ensure_ascii=False, indent=2)
        print(f"Erfolgreich {output_filename} erstellt.")
    except IOError as e:
        print(f"Fehler beim Schreiben der Datei {output_filename}: {e}")

if __name__ == "__main__":
    main()
