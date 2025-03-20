#date: 2025-03-20T17:10:03Z
#url: https://api.github.com/gists/b7cf9801e80e77d0827de561067b667c
#owner: https://api.github.com/users/me-suzy

import os
import re
from pathlib import Path
import unidecode
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError, TranslationNotFound

input_dir = r"e:\Carte\BB\17 - Site Leadership\alte\Ionel Balauta\Aryeht\Task 1 - Traduce tot site-ul\Doar Google Web\Andreea\Meditatii\2023\Iulia Python\output"
ro_dir = r"e:\Carte\BB\17 - Site Leadership\Principal\ro"
category_dir = r"e:\Carte\BB\17 - Site Leadership\Principal\en"

def translate_month(date_str):
    ro_to_en = {
        'Ianuarie': 'January', 'Februarie': 'February', 'Martie': 'March', 'Aprilie': 'April',
        'Mai': 'May', 'Iunie': 'June', 'Iulie': 'July', 'August': 'August',
        'Septembrie': 'September', 'Octombrie': 'October', 'Noiembrie': 'November', 'Decembrie': 'December'
    }
    for ro, en in ro_to_en.items():
        if ro in date_str:
            return date_str.replace(ro, en)
    return date_str

def calculate_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    text1 = unidecode.unidecode(text1.lower())
    text2 = unidecode.unidecode(text2.lower())
    words1 = set(re.findall(r'\b\w+\b', text1))
    words2 = set(re.findall(r'\b\w+\b', text2))
    if not words1 or not words2:
        return 0
    common_words = words1.intersection(words2)
    return len(common_words) / max(len(words1), len(words2))

def get_category_mapping():
    category_map = {
        'principiile-conducerii': {'link': 'leadership-principles', 'title': 'Leadership Principles'},
        'leadership-real': {'link': 'real-leadership', 'title': 'Real Leadership'},
        'legile-conducerii': {'link': 'leadership-laws', 'title': 'Leadership Laws'},
        'dezvoltare-personala': {'link': 'personal-development', 'title': 'Personal Development'},
        'leadership-de-succes': {'link': 'successful-leadership', 'title': 'Successful Leadership'},
        'lideri-si-atitudine': {'link': 'leadership-and-attitude', 'title': 'Leadership and Attitude'},
        'aptitudini-si-abilitati-de-leadership': {'link': 'leadership-skills-and-abilities', 'title': 'Leadership Skills And Abilities'},
        'hr-resurse-umane': {'link': 'hr-human-resources', 'title': 'Human Resources'},
        'leadership-total': {'link': 'total-leadership', 'title': 'Total Leadership'},
        'leadership-de-durata': {'link': 'leadership-that-lasts', 'title': 'Leadership That Lasts'},
        'calitatile-unui-lider': {'link': 'qualities-of-a-leader', 'title': 'Qualities of A Leader'},
        'leadership-de-varf': {'link': 'top-leadership', 'title': 'Top Leadership'},
        'jurnal-de-leadership': {'link': 'leadership-journal', 'title': 'Leadership Journal'}
    }
    return category_map, {v['link']: {'link': k, 'title': k.replace('-', ' ').title()} for k, v in category_map.items()}

def translate_category_link(link, title):
    category_map, _ = get_category_mapping()
    if link in category_map:
        return category_map[link]['link'], category_map[link]['title']
    return link, title

def extract_title(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        match = re.search(r'<h1 class="den_articol" itemprop="name">(.*?)</h1>', content)
        return match.group(1) if match else ''
    except Exception:
        return ''

def extract_description(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        match = re.search(r'<meta name="description" content="([^"]+)"', content)
        return match.group(1) if match else None
    except Exception:
        return None

def get_first_sentence(text):
    if not text:
        return ""
    sentences = re.split(r'[.!?]+', text)
    return sentences[0].strip() if sentences else text.strip()

def translate_text(text):
    try:
        return GoogleTranslator(source='ro', target='en').translate(text)
    except (RequestError, TranslationNotFound):
        return None

def build_ro_index(directory):
    index = {}
    for ro_file in Path(directory).glob('*.html'):
        content = open(ro_file, 'r', encoding='utf-8', errors='ignore').read()
        index[ro_file] = {
            'title': extract_title(ro_file),
            'desc_first': get_first_sentence(extract_description(ro_file) or ""),
            'content': content
        }
    return index

def find_real_ro_correspondent(en_file_path, ro_index):
    en_filename = Path(en_file_path).stem
    print(f"Căutare corespondent pentru: {en_filename}")

    with open(en_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        en_content = f.read()
    en_title = extract_title(en_file_path)
    en_desc_first = get_first_sentence(extract_description(en_file_path) or "")

    # Pasul 1: Similitudine titlu
    best_match = None
    best_score = 0
    for ro_file, data in ro_index.items():
        if data['title'] and en_title:
            score = calculate_similarity(en_title, data['title'])
            if score > best_score:
                best_score = score
                best_match = ro_file
            if score > 0.45:
                print(f"Potrivire titlu: {ro_file.name} (Scor: {score:.2f})")
    if best_match and best_score > 0.45:
        print(f"Corespondent găsit prin titlu: {best_match.name} (Scor: {best_score:.2f})")
        return str(best_match)

    # Pasul 2: Similitudine descriere
    if en_desc_first:
        best_match = None
        best_score = 0
        for ro_file, data in ro_index.items():
            if data['desc_first']:
                translated_desc = translate_text(data['desc_first'])
                if translated_desc:
                    score = calculate_similarity(en_desc_first, translated_desc)
                    if score > best_score:
                        best_score = score
                        best_match = ro_file
                    if score > 0.7:
                        print(f"Potrivire descriere: {ro_file.name} (Scor: {score:.2f})")
        if best_match and best_score > 0.7:
            print(f"Corespondent găsit prin descriere: {best_match.name} (Scor: {best_score:.2f})")
            return str(best_match)

    # Pasul 3: Referințe încrucișate (cu verificare)
    for ro_file, data in ro_index.items():
        if f"en/{en_filename}.html" in data['content']:
            # Verificăm dacă titlul sau descrierea confirmă
            title_score = calculate_similarity(en_title, data['title']) if en_title and data['title'] else 0
            desc_score = calculate_similarity(en_desc_first, translate_text(data['desc_first']) or "") if en_desc_first and data['desc_first'] else 0
            if title_score > 0.3 or desc_score > 0.5:
                print(f"Corespondent găsit prin referință RO->EN confirmată: {ro_file.name}")
                return str(ro_file)
        if f"https://neculaifantanaru.com/{ro_file.stem}.html" in en_content:
            title_score = calculate_similarity(en_title, data['title']) if en_title and data['title'] else 0
            desc_score = calculate_similarity(en_desc_first, translate_text(data['desc_first']) or "") if en_desc_first and data['desc_first'] else 0
            if title_score > 0.3 or desc_score > 0.5:
                print(f"Corespondent găsit prin referință EN->RO confirmată: {ro_file.name}")
                return str(ro_file)

    print(f"Nu s-a găsit corespondent pentru {en_filename}")
    return None

def update_flags_section(en_file_path, ro_file_path):
    en_filename = Path(en_file_path).stem
    ro_filename = Path(ro_file_path).stem
    with open(en_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        en_content = f.read()

    current_ro_link = re.search(r'<a href="https://neculaifantanaru\.com/([^"]+)\.html"><img src="index_files/flag_lang_ro\.jpg"', en_content)
    current_ro_link = current_ro_link.group(1) if current_ro_link else None
    if current_ro_link == ro_filename:
        return False

    flags_match = re.search(r'(<!-- FLAGS_1 -->.*?<!-- FLAGS -->)', en_content, re.DOTALL)
    if not flags_match:
        return False

    old_flags = flags_match.group(1)
    new_flags = re.sub(
        r'<a href="https://neculaifantanaru\.com/([^"]+)\.html"><img src="index_files/flag_lang_ro\.jpg"',
        f'<a href="https://neculaifantanaru.com/{ro_filename}.html"><img src="index_files/flag_lang_ro.jpg"',
        old_flags
    )
    new_flags = re.sub(
        r'<a href="https://neculaifantanaru\.com/en/([^"]+)\.html"><img src="index_files/flag_lang_en\.jpg"',
        f'<a href="https://neculaifantanaru.com/en/{en_filename}.html"><img src="index_files/flag_lang_en.jpg"',
        new_flags
    )

    if new_flags == old_flags:
        return False

    updated_content = en_content.replace(old_flags, new_flags)
    with open(en_file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    print(f"FLAGS actualizat: EN={en_filename}, RO={ro_filename}")
    return True

def extract_category_info(ro_file):
    with open(ro_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    match = re.search(r'<td class="text_dreapta">On (.*?), in <a href="https://neculaifantanaru\.com/(.*?)\.html" title="Vezi toate articolele din (.*?)"', content)
    if match:
        link, title = translate_category_link(match.group(2), match.group(3))
        return {'date': translate_month(match.group(1)), 'category_link': link, 'category_title': title}
    return None

def update_category_file(category_file, article_data, quote):
    with open(category_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    if f'href="https://neculaifantanaru.com/en/{article_data["filename"]}"' in content:
        return False

    article_section = f'''      <table width="660" border="0">
            <tr><td><span class="den_articol"><a href="https://neculaifantanaru.com/en/{article_data['filename']}" class="linkMare">{article_data['title']}</a></span></td></tr>
            <tr><td class="text_dreapta">On {article_data['date']}, in <a href="https://neculaifantanaru.com/en/{article_data['category_link']}.html" title="View all articles from {article_data['category_title']}" class="external" rel="category tag">{article_data['category_title']}</a>, by Neculai Fantanaru</td></tr>
          </table>
          <p class="text_obisnuit2">{quote}</p>
          <table width="552" border="0">
            <tr><td width="552"><div align="right" id="external2"><a href="https://neculaifantanaru.com/en/{article_data['filename']}">read more </a><a href="https://neculaifantanaru.com/en/" title=""><img src="Arrow3_black_5x7.gif" alt="" width="5" height="7" class="arrow" /></a></div></td></tr>
          </table>
          <p class="text_obisnuit"></p>'''

    start_pos = content.find('<!-- ARTICOL CATEGORIE START -->')
    div_start = content.find('<div align="justify">', start_pos)
    end_pos = content.find('<!-- ARTICOL CATEGORIE FINAL -->')

    if start_pos != -1 and end_pos != -1 and div_start != -1:
        is_2024 = "2024" in article_data['date']
        if is_2024:
            div_end = div_start + len('<div align="justify">')
            new_content = content[:div_end] + '\n' + article_section + content[div_end:]
        else:
            section_content = content[start_pos:end_pos]
            last_table_pos = section_content.rfind('</table')
            if last_table_pos != -1:
                insert_position = start_pos + last_table_pos + len('</table>\n      <p class="text_obisnuit"></p>')
                new_content = (content[:insert_position] + '\n' + article_section +
                               '\n          </div>\n          <p align="justify" class="text_obisnuit style3"> </p>' + content[end_pos:])
        with open(category_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def process_files():
    print("Construire index RO...")
    ro_index = build_ro_index(ro_dir)
    print(f"Index construit cu {len(ro_index)} intrări.")

    files_processed = 0
    files_flags_updated = 0
    files_category_updated = 0
    files_skipped = 0
    files_with_errors = 0

    for en_file in Path(input_dir).glob('*.html'):
        print(f"\nProcesare: {en_file.name}")
        try:
            ro_file = find_real_ro_correspondent(en_file, ro_index)
            if not ro_file:
                files_skipped += 1
                continue

            print(f"Corespondent: {Path(ro_file).name}")
            if update_flags_section(en_file, ro_file):
                files_flags_updated += 1

            category_info = extract_category_info(ro_file)
            if not category_info:
                print(f"Nu s-au extras informații de categorie din {Path(ro_file).name}")
                files_skipped += 1
                continue

            quote = extract_quote(en_file)
            title = extract_title(en_file)
            category_info['filename'] = en_file.name
            category_info['title'] = title

            category_file = Path(category_dir) / f"{category_info['category_link']}.html"
            if update_category_file(category_file, category_info, quote):
                files_category_updated += 1

            files_processed += 1
        except Exception as e:
            print(f"Eroare: {e}")
            files_with_errors += 1

    print("\nRezumat:")
    print(f"Procesate: {files_processed}")
    print(f"FLAGS actualizate: {files_flags_updated}")
    print(f"Categorii actualizate: {files_category_updated}")
    print(f"Omise: {files_skipped}")
    print(f"Erori: {files_with_errors}")

def extract_quote(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        match = re.search(r'<p class="text_obisnuit2">(.*?)</p>', content)
        return match.group(1).strip() if match else ''
    except Exception:
        return ''

if __name__ == "__main__":
    process_files()