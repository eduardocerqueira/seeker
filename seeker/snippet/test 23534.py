#date: 2024-11-08T16:46:21Z
#url: https://api.github.com/gists/f27a3fe8f0c76cc519f354c7c369d004
#owner: https://api.github.com/users/me-suzy

from bs4 import BeautifulSoup
import re
from typing import List, Dict, Set

def print_html_structure(soup, tag, level=0):
    """Printează structura ierarhică a HTML-ului."""
    indent = "  " * level
    attrs = ' '.join([f'{k}="{v}"' for k, v in tag.attrs.items()])
    print(f"{indent}{tag.name} {attrs}")
    for child in tag.children:
        if child.name:  # skip NavigableString objects
            print_html_structure(soup, child, level + 1)

def get_tag_structure(tag):
    """Returnează o reprezentare string a structurii unui tag."""
    if not tag:
        return ""
    attrs = ' '.join([f'{k}="{v}"' for k, v in sorted(tag.attrs.items())])
    structure = f"{tag.name} {attrs}".strip()

    child_structures = []
    for child in tag.children:
        if child.name:
            child_structures.append(get_tag_structure(child))

    if child_structures:
        structure += " > " + " > ".join(child_structures)

    return structure

def extract_article_content(html_content: str) -> BeautifulSoup:
    """Extrage conținutul dintre marcajele ARTICOL."""
    start = "<!-- ARTICOL START -->"
    end = "<!-- ARTICOL FINAL -->"
    start_idx = html_content.find(start)
    end_idx = html_content.find(end) + len(end)

    if start_idx == -1 or end_idx == -1:
        raise ValueError("Nu s-au găsit marcajele ARTICOL START/FINAL")

    article_content = html_content[start_idx:end_idx]
    return BeautifulSoup(article_content, 'html.parser')

def analyze_tag_structure(soup: BeautifulSoup):
    """Analizează și printează structura tuturor tagurilor relevante."""
    print("\nAnaliză structură HTML:")
    print("=" * 50)

    for tag in soup.find_all(['p']):
        print("\nTag găsit:")
        print(f"Structură completă: {get_tag_structure(tag)}")
        print(f"Text conținut: {tag.get_text(strip=True)}")
        print(f"Atribute: {tag.attrs}")
        if tag.find('span'):
            print("Conține span:")
            for span in tag.find_all('span'):
                print(f"  Span atribute: {span.attrs}")
                print(f"  Span text: {span.get_text(strip=True)}")

def find_matching_structure(ro_tag, en_tags, used_en_indices):
    """Găsește tag-ul EN care se potrivește cel mai bine structural cu tag-ul RO."""
    ro_structure = get_tag_structure(ro_tag)
    best_match = None
    best_score = float('inf')

    print(f"\nCăutare potrivire pentru RO tag: {ro_structure}")
    print(f"Text RO: {ro_tag.get_text(strip=True)}")

    for i, en_tag in enumerate(en_tags):
        if i in used_en_indices:
            continue

        en_structure = get_tag_structure(en_tag)
        print(f"\n  Comparare cu EN tag #{i}: {en_structure}")
        print(f"  Text EN: {en_tag.get_text(strip=True)}")

        # Verifică dacă structurile se potrivesc
        if ro_structure == en_structure:
            # Calculează scorul bazat pe diferența de lungime text
            ro_words = len(ro_tag.get_text(strip=True).split())
            en_words = len(en_tag.get_text(strip=True).split())
            word_diff = abs(ro_words - en_words)

            print(f"  Structuri identice! Diferență cuvinte: {word_diff}")

            if word_diff < best_score:
                best_score = word_diff
                best_match = i
                print(f"  Nou best match găsit: #{i}")

    if best_match is not None and best_score <= 3:
        return best_match
    return None

def main():
    try:
        # Definim căile către fișiere
        ro_file_path = r'd:\3\ro\incotro-vezi-tu-privire.html'
        en_file_path = r'd:\3\en\where-do-you-see-look.html'
        output_file_path = r'd:\3\Output\where-do-you-see-look.html'

        # Citim fișierele
        with open(ro_file_path, 'r', encoding='utf-8') as f:
            ro_content = f.read()
            print("\nConținut RO încărcat:")
            print(ro_content[:200] + "...")

        with open(en_file_path, 'r', encoding='utf-8') as f:
            en_content = f.read()
            print("\nConținut EN încărcat:")
            print(en_content[:200] + "...")

        # Parsăm conținutul
        ro_soup = extract_article_content(ro_content)
        en_soup = extract_article_content(en_content)

        print("\nAnaliză structură document RO:")
        analyze_tag_structure(ro_soup)

        print("\nAnaliză structură document EN:")
        analyze_tag_structure(en_soup)

        # Găsim toate tagurile relevante
        ro_tags = ro_soup.find_all(['p'])
        en_tags = en_soup.find_all(['p'])

        print(f"\nTotal taguri găsite - RO: {len(ro_tags)}, EN: {len(en_tags)}")

        # Procesăm potrivirile
        matches = {}
        used_en_indices = set()

        # Prima trecere: găsim potriviri exacte structural
        for ro_idx, ro_tag in enumerate(ro_tags):
            en_idx = find_matching_structure(ro_tag, en_tags, used_en_indices)
            if en_idx is not None:
                matches[ro_idx] = en_idx
                used_en_indices.add(en_idx)
                print(f"\nPotrivire găsită: RO #{ro_idx} -> EN #{en_idx}")

        # Generăm output-ul
        output_content = ["<!-- ARTICOL START -->"]

        # Adăugăm tagurile în ordinea corectă
        for ro_idx, ro_tag in enumerate(ro_tags):
            if ro_idx in matches:
                # Folosim versiunea EN
                output_content.append(str(en_tags[matches[ro_idx]]))
                print(f"\nFolosit tag EN #{matches[ro_idx]} pentru RO #{ro_idx}")
            else:
                # Păstrăm versiunea RO
                output_content.append(str(ro_tag))
                print(f"\nPăstrat tag RO #{ro_idx} (fără potrivire)")

        # Adăugăm tagurile EN rămase
        for en_idx, en_tag in enumerate(en_tags):
            if en_idx not in used_en_indices:
                output_content.append(str(en_tag))
                print(f"\nAdăugat tag EN #{en_idx} rămas")

        output_content.append("<!-- ARTICOL FINAL -->")
        final_output = "\n".join(output_content)

        # Scriem rezultatul
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(final_output)

        print(f"\nOutput final generat în: {output_file_path}")
        print("\nConținut output:")
        print(final_output)

    except Exception as e:
        print(f"Eroare: {str(e)}")
        raise

if __name__ == "__main__":
    main()