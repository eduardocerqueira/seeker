#date: 2025-04-18T16:28:39Z
#url: https://api.github.com/gists/16a1c07a12f1ea761a579a527bdc2981
#owner: https://api.github.com/users/me-suzy

#!/usr/bin/env python3
import requests
import re
import os
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# Configurare pentru Anthropic Claude API
CLAUDE_API_KEY = "YOUR-API-KEY"  # Înlocuiește cu cheia ta reală Claude API
CLAUDE_API_ENDPOINT = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-3-opus-20240229"  # Poți folosi 'claude-3-sonnet' pentru procesare mai rapidă

# Variabilă globală pentru a stoca dacă API-ul este disponibil
use_api = True if CLAUDE_API_KEY != "YOUR-API-KEY" else False

# Configurare pentru logging detaliat
VERBOSE = True

def log_info(message):
    """Afișează un mesaj de informare cu timestamp."""
    if VERBOSE:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] INFO: {message}")

def log_warning(message):
    """Afișează un mesaj de avertizare cu timestamp."""
    if VERBOSE:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] AVERTISMENT: {message}")

def log_error(message):
    """Afișează un mesaj de eroare cu timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] EROARE: {message}")

def read_local_file(filepath):
    """Citește textul din fișierul local."""
    log_info(f"Încercăm să citim fișierul local: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            log_info(f"Fișierul local a fost citit cu succes: {len(content)} caractere.")
            return content
    except Exception as e:
        log_error(f"Eroare la citirea fișierului local: {e}")
        return None

def extract_personalities(text):
    """Extrage personalitățile din text folosind un pattern matcher îmbunătățit."""
    log_info("Începem extragerea personalităților din text...")

    # Pattern principal pentru identificarea personalităților
    pattern = r'^([A-ZĂÂÎȘȚ][A-ZĂÂÎȘȚ\s\-]+)\s*\(([^)]+)\)\s*([\w\s\-]+)\s*'
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    log_info(f"Am găsit {len(matches)} potențiale personalități în text.")

    personalities = []
    for i, match in enumerate(matches):
        name = match.group(1).strip().title()
        years = match.group(2).strip()
        profession = match.group(3).strip()

        # Parsăm anii și datele exacte
        birth_date = None
        death_date = None
        birth_year = None
        death_year = None
        birth_place = None
        death_place = None
        education = []
        distinctions = []
        works = []
        images = []

        # Extragem anii
        years_match = re.match(r'(\d{4})\s*-\s*(\d{4})?', years)
        if years_match:
            birth_year = years_match.group(1)
            death_year = years_match.group(2) if years_match.group(2) else None
        elif "n." in years:
            birth_year = years.replace("n.", "").strip()

        # Extragem textul biografic
        start_pos = match.end()
        end_pos = matches[i+1].start() if i < len(matches)-1 else len(text)
        biography = text[start_pos:end_pos].strip()

        # Extragem data și locul nașterii
        birth_match = re.search(r'(?:s-a născut|născut[ă]?) (?:la|pe) (\d{1,2}\s+\w+\s+\d{4})[,\s]+(?:în|la)\s+([^\.]+)', biography)
        if birth_match:
            birth_date = birth_match.group(1)
            birth_place = birth_match.group(2).strip().rstrip('.')

        # Extragem data și locul decesului
        death_match = re.search(r'(?:a murit|decedat[ă]?) (?:la|pe) (\d{1,2}\s+\w+\s+\d{4})[,\s]+(?:în|la)\s+([^\.]+)', biography)
        if death_match:
            death_date = death_match.group(1)
            death_place = death_match.group(2).strip().rstrip('.')

        # Extragem educația
        edu_matches = re.finditer(r'(?:studiat|absolvit)\s+(?:la|în)?\s*([\w\s,]+?)(?:[\.\,\;]|\s+(?:unde|și|\d{4}))', biography)
        for edu_match in edu_matches:
            education.append(edu_match.group(1).strip())

        # Extragem distincțiile
        dist_matches = re.finditer(r'(?:Premiul|Membru|Ordinul|Doctor|Distincția|Magna\s+cum\s+Laude)\s+([^\.\;]+)', biography)
        for dist_match in dist_matches:
            distinctions.append(dist_match.group(1).strip())

        # Extragem lucrările publicate
        work_matches = re.finditer(r'(?:„|")([^„"]+)(?:”|")\s*(?:\([^)]+\))?(?:,\s*[\w\s]+,\s*\d{4})?', biography)
        for work_match in work_matches:
            works.append(work_match.group(1).strip())

        # Extragem imaginile asociate lucrărilor
        img_matches = re.finditer(r'([A-Z][a-zăâîșț\s\-]+ - [A-Za-z\s\-\.]+)\.jpg', biography)
        for img_match in img_matches:
            images.append(img_match.group(1) + '.jpg')

        # Curățăm textul biografic
        biography = re.sub(r'\s+', ' ', biography)  # Înlocuim spațiile multiple
        biography = biography.replace('~', '-')     # Înlocuim caracterele problematice

        log_info(f"Personalitate extrasă: {name} ({years}) - {profession} | Date: {birth_date}, {death_date} | Text: {len(biography)} caractere")

        personalities.append({
            'name': name,
            'years': f"{birth_year}-{death_year}" if birth_year and death_year else f"n. {birth_year}" if birth_year else years,
            'birth_date': birth_date,
            'birth_year': birth_year,
            'death_date': death_date,
            'death_year': death_year,
            'birth_place': birth_place,
            'death_place': death_place,
            'profession': profession,
            'education': education,
            'distinctions': distinctions,
            'works': works,
            'images': images,
            'bio_text': biography
        })

    log_info(f"Extragere completă: {len(personalities)} personalități identificate.")
    return personalities

def call_claude_api(prompt: "**********": str, max_tokens: int = 1000, temperature: float = 0.7) -> Optional[str]:
    """Funcție pentru apeluri către Anthropic Claude API."""
    if not use_api:
        log_warning("API-ul Claude nu este disponibil. Se continuă fără procesare AI.")
        return None

    log_info(f"Trimit cerere către Claude API - prompt: "**********": {max_tokens}")

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }

    payload = {
        "model": CLAUDE_MODEL,
        "system": system_message,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": "**********"
        "temperature": temperature
    }

    try:
        log_info("Așteaptă răspuns de la API...")
        start_time = time.time()
        response = requests.post(CLAUDE_API_ENDPOINT, headers=headers, json=payload)
        elapsed_time = time.time() - start_time
        log_info(f"Răspuns primit în {elapsed_time:.2f} secunde.")

        if response.status_code != 200:
            log_error(f"Cererea API a eșuat: {response.status_code} - {response.text}")
            return None

        result = response.json()
        response_text = result["content"][0]["text"]
        log_info(f"Răspuns procesat: {len(response_text)} caractere.")
        return response_text
    except Exception as e:
        log_error(f"Eroare în apelul API: {e}")
        return None

def detect_and_translate_text(person: Dict[str, Any]) -> Dict[str, Any]:
    """Detectează limba și traduce textul în română dacă este necesar."""
    if not use_api:
        log_warning(f"API indisponibil, se păstrează textul original pentru {person['name']}.")
        return person

    log_info(f"Detectez limba pentru {person['name']}...")
    if len(person["bio_text"]) < 100:
        log_warning(f"Biografie prea scurtă pentru {person['name']}, se sare peste traducere.")
        return person

    sample_text = person["bio_text"][:500]
    prompt = f"""
    Determină limba textului următor și, dacă nu este română, traduce textul complet în română, păstrând termenii tehnici și numele proprii neschimbate.
    Text: "{sample_text}..."
    Răspunde în JSON:
    {{
        "limba_detectata": "română/engleză/etc.",
        "necesita_traducere": true/false,
        "text_tradus": ""
    }}
    """
    system_message = "Ești expert în detectarea limbilor și traducere precisă."
    detection_result = "**********"=800, temperature=0.3)

    if not detection_result:
        log_warning(f"Nu s-a detectat limba pentru {person['name']}, se păstrează originalul.")
        return person

    try:
        parsed_result = json.loads(detection_result)
        detected_language = parsed_result.get("limba_detectata", "")
        needs_translation = parsed_result.get("necesita_traducere", False)

        log_info(f"Limba detectată pentru {person['name']}: {detected_language}")
        if needs_translation:
            log_info(f"Traducere necesară pentru {person['name']}...")
            full_text_prompt = f"""
            Traduce din {detected_language} în română textul următor, păstrând termenii tehnici și numele proprii:
            {person["bio_text"]}
            """
            system_message = "Ești expert în traducere. Păstrează sensul original și formatarea."
            full_translation = "**********"=3000, temperature=0.3)
            if full_translation:
                person["bio_text"] = full_translation
                log_info(f"Biografie tradusă pentru {person['name']}.")
            else:
                log_warning(f"Traducere eșuată pentru {person['name']}, se păstrează originalul.")
        else:
            log_info(f"Textul pentru {person['name']} este deja în română.")
    except json.JSONDecodeError:
        log_warning(f"Eroare la parsarea JSON pentru {person['name']}, se păstrează originalul.")
    return person

def generate_comprehensive_article(person: Dict[str, Any]) -> Dict[str, Any]:
    """Generează un articol Wikipedia detaliat."""
    if not use_api:
        log_warning(f"API indisponibil, se generează articol de bază pentru {person['name']}.")
        person["article_content"] = f"""
'''{person['name']}''' ({person['years']}) a fost un {person['profession']} român.

== Biografie ==
{person['bio_text']}

== Activitate științifică ==
{person['name']} a adus contribuții în domeniul {person['profession'].lower()}.

== Note ==
<references />

== Bibliografie ==
* Pestean, Viorel Iulian - ''Oameni de seamă ai științei agricole românești'' (Vol.2)
"""
        return person

    log_info(f"Generez articol pentru {person['name']}...")

    # Construim promptul cu toate datele extrase
    works_str = "\n".join([f"* „{work}”" for work in person['works']]) if person['works'] else "Necunoscute"
    distinctions_str = "\n".join([f"* {dist}" for dist in person['distinctions']]) if person['distinctions'] else "Necunoscute"
    images_str = "\n".join([f"* [[File:{img}|thumb]]" for img in person['images']]) if person['images'] else "Niciuna"

    prompt = f"""
Creează un articol Wikipedia detaliat în română despre {person['name']} ({person['years']}), un {person['profession']}.

Folosește informațiile:
- Biografie: {person['bio_text']}
- Data nașterii: {person.get('birth_date', 'Necunoscută')}
- Locul nașterii: {person.get('birth_place', 'Necunoscut')}
- Data decesului: {person.get('death_date', 'Necunoscută')}
- Locul decesului: {person.get('death_place', 'Necunoscut')}
- Educație: {', '.join(person['education']) if person['education'] else 'Necunoscută'}
- Distincții:
{distinctions_str}
- Lucrări:
{works_str}
- Imagini asociate lucrărilor:
{images_str}

Structura articolului:
1. Rezumat biografic
2. == Biografie ==
   === Origini și educație ===
   === Cariera profesională ===
3. == Activitate științifică și publicistică ==
   === Cercetări ===
   === Lucrări majore === (include imaginile aici, fără descrieri)
   === Contribuții administrative ===
4. == Distincții ==
5. == Lucrări publicate ==
6. == Note ==
7. == Bibliografie ==
   * Include: Pestean, Viorel Iulian - ''Oameni de seamă ai științei agricole românești'' (Vol.2)
8. == Legături externe ==

Reguli:
- Stil enciclopedic, obiectiv
- Nu inventa informații
- Include toate datele furnizate
- Imaginile se plasează în „Lucrări majore” cu formatul [[File:Nume_Imagine.jpg|thumb]]
- Secțiunile goale pot fi omise
"""
    system_message = "Ești expert în crearea articolelor Wikipedia în română, respectând formatarea MediaWiki."
    article_content = "**********"=4000, temperature=0.5)

    if article_content:
        person["article_content"] = article_content
        log_info(f"Articol generat pentru {person['name']} ({len(article_content)} caractere).")
    else:
        log_error(f"Eșec generare articol pentru {person['name']}, se folosește șablon de bază.")
        person["article_content"] = f"""
'''{person['name']}''' ({person['years']}) a fost un {person['profession']} român.

== Biografie ==
{person['bio_text']}

== Activitate științifică ==
{person['name']} a adus contribuții în domeniul {person['profession'].lower()}.

== Note ==
<references />

== Bibliografie ==
* Pestean, Viorel Iulian - ''Oameni de seamă ai științei agricole românești'' (Vol.2)
"""
    return person

def generate_infobox(person: Dict[str, Any]) -> str:
    """Generează infocasetă Wikipedia."""
    log_info(f"Creez infocasetă pentru {person['name']}...")
    birth_date = person.get('birth_date', person.get('birth_year', ''))
    death_date = person.get('death_date', person.get('death_year', ''))
    birth_place = person.get('birth_place', '')
    death_place = person.get('death_place', '')
    education = ', '.join(person['education']) if person['education'] else ''
    distinctions = '; '.join(person['distinctions']) if person['distinctions'] else ''
    works = '; '.join([f"„{w}”" for w in person['works']]) if person['works'] else ''

    infobox = f"""{{{{Infocaseta Biografie
| nume            = {person['name']}
| imagine         = {person['name'].replace(' ', '_')}.jpg
| mărime_imagine  =
| descriere_imagine =
| data_nașterii   = {birth_date}
| loc_naștere     = {birth_place}
| data_decesului  = {death_date}
| loc_deces       = {death_place}
| naționalitate   = Română
| educație        = {education}
| ocupație        = {person['profession']}
| cunoscut_pentru = Contribuții în {person['profession'].lower()}
| lucrări_importante = {works}
| premii          = {distinctions}
}}}}"""
    return infobox

def generate_categories(person: Dict[str, Any]) -> List[str]:
    """Generează categorii Wikipedia."""
    log_info(f"Generez categorii pentru {person['name']}...")
    categories = []
    profession_lower = person['profession'].lower()

    if person.get('birth_year'):
        categories.append(f"[[Categorie:Nașteri în {person['birth_year']}]]")
    if person.get('death_year'):
        categories.append(f"[[Categorie:Decese în {person['death_year']}]]")

    if "pedolog" in profession_lower:
        categories.append("[[Categorie:Pedologi români]]")
    if "agronom" in profession_lower:
        categories.append("[[Categorie:Agronomi români]]")
    if "zootehnist" in profession_lower:
        categories.append("[[Categorie:Zootehniști români]]")
    if "profesor" in profession_lower:
        categories.append("[[Categorie:Profesori universitari români]]")
    if "chimist" in profession_lower:
        categories.append("[[Categorie:Chimiști români]]")
    if "academiei române" in person.get('distinctions', []):
        categories.append("[[Categorie:Membri ai Academiei Române]]")

    categories.append("[[Categorie:Oameni de știință români]]")
    log_info(f"{len(categories)} categorii pentru {person['name']}.")
    return categories

def format_wikipedia_article(person: Dict[str, Any]) -> str:
    """Formatează articolul Wikipedia final."""
    log_info(f"Formatez articol pentru {person['name']}...")
    infobox = generate_infobox(person)
    article_content = person.get("article_content", f"""
'''{person['name']}''' ({person['years']}) a fost un {person['profession']} român.

== Biografie ==
{person['bio_text']}

== Activitate științifică ==
{person['name']} a adus contribuții în domeniul {person['profession'].lower()}.

== Note ==
<references />

== Bibliografie ==
* Pestean, Viorel Iulian - ''Oameni de seamă ai științei agricole românești'' (Vol.2)
""")
    categories = generate_categories(person)
    defaultsort = f"{{{{DEFAULTSORT:{person['name'].split()[-1]}, {' '.join(person['name'].split()[:-1])}}}}}"

    # Adăugăm imaginile în „Lucrări majore” dacă nu sunt deja incluse
    if person['images'] and "== Lucrări majore ===" in article_content:
        for img in person['images']:
            if f"[[File:{img}|thumb]]" not in article_content:
                article_content = article_content.replace("== Lucrări majore ===", f"== Lucrări majore ===\n[[File:{img}|thumb]]")
    elif person['images'] and "== Lucrări majore ===" not in article_content:
        article_content += f"\n\n== Lucrări majore ===\n" + "\n".join([f"[[File:{img}|thumb]]" for img in person['images']])

    full_article = f"{infobox}\n\n{article_content}\n\n{defaultsort}\n{chr(10).join(categories)}"
    log_info(f"Articol formatat pentru {person['name']}: {len(full_article)} caractere.")
    return full_article

def save_wikipedia_article(name, content, output_dir="wikipedia_articles"):
    """Salvează articolul Wikipedia într-un fișier."""
    log_info(f"Salvez articolul pentru {name}...")
    Path(output_dir).mkdir(exist_ok=True)
    safe_name = re.sub(r'[^\w\s]', '', name).replace(" ", "_")
    filepath = os.path.join(output_dir, f"{safe_name}.txt")

    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        log_info(f"Articol salvat: {filepath}")
        return True
    except Exception as e:
        log_error(f"Eroare la salvarea articolului pentru {name}: {e}")
        return False

def main():
    print("=== Generator Articole Wikipedia pentru Oameni de Știință Agricolă Români ===")
    if not use_api:
        log_warning("Cheia Claude API lipsește. Articolele vor fi generate fără AI.")

    # Sursa datelor
    local_filepath = r"e:\Carte\BB\17 - Site Leadership\alte\Ionel Balauta\Aryeht\Task 1 - Traduce tot site-ul\Doar Google Web\Andreea\Meditatii\2023\Wikipedia\Pestean, Viorel Iulian - Oameni de seama ai stiintei agricole romanesti (Vol.2)_djvu.txt"
    url = "https://archive.org/stream/pestean-viorel-iulian-oameni-de-seama-ai-stiintei-agricole-romanesti-vol.-2/Pestean%2C%20Viorel%20Iulian%20-%20Oameni%20de%20seama%20ai%20stiintei%20agricole%20romanesti%20%28Vol.2%29_djvu.txt"

    log_info("Citim textul sursă...")
    text = read_local_file(local_filepath)
    if not text:
        log_warning("Fișierul local nu a fost găsit, se încearcă online...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            text = response.text
            log_info(f"Text descărcat online: {len(text)} caractere.")
        except Exception as e:
            log_error(f"Eroare la descărcare: {e}")
            return

    log_info(f"Text încărcat: {len(text)} caractere.")
    personalities = extract_personalities(text)

    if not personalities:
        log_error("Nicio personalitate găsită.")
        return

    log_info(f"{len(personalities)} personalități găsite.")
    translated_count = 0
    article_count = 0
    successful_count = 0

    for i, person in enumerate(personalities):
        print(f"\n{'='*80}")
        log_info(f"Procesăm {i+1}/{len(personalities)}: {person['name']}")
        print(f"{'='*80}")

        if len(person.get("bio_text", "").strip()) < 50:
            log_warning(f"Se sare peste {person['name']} - text insuficient.")
            continue

        if use_api:
            try:
                person = detect_and_translate_text(person)
                translated_count += 1
                person = generate_comprehensive_article(person)
                article_count += 1
            except Exception as e:
                log_error(f"Eroare API pentru {person['name']}: {e}")

        article = format_wikipedia_article(person)
        if save_wikipedia_article(person['name'], article):
            successful_count += 1

        if use_api:
            time.sleep(1)  # Evită limitările API

    print("\n"+"="*80)
    log_info("=== Sumar ===")
    log_info(f"Personalități găsite: {len(personalities)}")
    log_info(f"Traduceri: {translated_count}")
    log_info(f"Articole generate: {article_count}")
    log_info(f"Articole salvate: {successful_count}")
    log_info("Procesare completă!")

if __name__ == "__main__":
    main()erate: {article_count}")
    log_info(f"Articole salvate: {successful_count}")
    log_info("Procesare completă!")

if __name__ == "__main__":
    main()