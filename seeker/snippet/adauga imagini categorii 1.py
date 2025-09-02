#date: 2025-09-02T17:13:13Z
#url: https://api.github.com/gists/aac5c442ac2b1bc4bd2bc03d21550187
#owner: https://api.github.com/users/me-suzy

import os
import requests
from bs4 import BeautifulSoup, Comment
import time
import re

def get_image_from_article_page(url):
    """Extrage imaginea din pagina individuală a articolului"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Caută imaginea în feature-img-wrap
        feature_img = soup.find('div', class_='feature-img-wrap')
        if feature_img:
            img = feature_img.find('img')
            if img and img.get('src'):
                return img.get('src'), img.get('alt', '')

        # Fallback - caută orice img cu src care conține 'images/'
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if 'images/' in src and not src.endswith('.gif'):
                return src, img.get('alt', '')

        return None, None

    except Exception as e:
        print(f"Eroare la extragerea imaginii din {url}: {e}")
        return None, None

def create_new_article_html(title, url, date, category, category_url, author, description, img_src, img_alt):
    """Creează noul HTML pentru articol cu layout desktop și mobil"""

    return f'''<article class="blog-box heading-space-half">
    <div class="blog-listing-inner news_item">
        <div class="article-card-new">

            <!-- Layout DESKTOP - ascuns pe mobil -->
            <div class="desktop-layout d-none d-md-block">
                <div class="article-header d-flex">
                    <div class="article-image-container">
                        <a href="{url}">
                            <img src="{img_src}"
                                 alt="{img_alt}"
                                 class="article-card-img">
                        </a>
                    </div>

                    <div class="article-header-content">
                        <h2 class="custom-h1" itemprop="name">
                            <a href="{url}" class="color-black">{title}</a>
                        </h2>

                        <div class="article-spacing"></div>
                        <div class="article-spacing"></div>

                        <div class="blog-post d-flex align-items-center flex-wrap">
                            <i class="fa fa-calendar mx-1"></i>
                            <time datetime="{date.split()[-1]}" class="color-black">{date}, in</time>
                            <a href="{category_url}" class="color-green font-weight-600 mx-1">{category}</a>
                        </div>
                        <div class="author-info color-black">by {author}</div>
                    </div>
                </div>

                <div class="article-body">
                    <div class="article-spacing"></div>
                    <div class="article-spacing"></div>

                    <p class="color-grey line-height-25px">{description}</p>

                    <a href="{url}" class="btn-setting color-black btn-hvr-up btn-blue btn-hvr-pink">
                        read more<span class="sr-only"> despre {title}</span>
                    </a>
                </div>
            </div>

            <!-- Layout MOBIL - afișat doar pe mobil -->
            <div class="mobile-layout d-block d-md-none">
                <div class="mobile-image-container">
                    <a href="{url}">
                        <img src="{img_src}"
                             alt="{img_alt}"
                             class="mobile-article-img">
                    </a>
                </div>

                <h2 class="custom-h1 mobile-title">
                    <a href="{url}" class="color-black">{title}</a>
                </h2>

                <p class="color-grey line-height-25px mobile-lead">{description}</p>

                <div class="blog-post mobile-date">
                    <i class="fa fa-calendar mx-1"></i>
                    <time datetime="{date.split()[-1]}" class="color-black">{date}, in</time>
                    <a href="{category_url}" class="color-green font-weight-600 mx-1">{category}</a>
                </div>

                <a href="{url}" class="btn-setting color-black btn-hvr-up btn-blue btn-hvr-pink mobile-read-more">
                    Read More<span class="sr-only"> despre {title}</span>
                </a>
            </div>

        </div>
    </div>
</article>'''

def get_article_css():
    """Returnează CSS-ul pentru articole"""
    return '''
    <style type="text/css">
        /* CSS pentru layout articole */
        .article-card-new {
            padding: 15px;
            margin-bottom: 20px;
            border-bottom: 1px solid #eee;
        }

        /* Stiluri pentru DESKTOP */
        .desktop-layout .article-header {
            display: flex;
            gap: 20px;
            align-items: flex-start;
        }

        .desktop-layout .article-image-container {
            flex-shrink: 0;
            width: 180px;
        }

        .desktop-layout .article-card-img {
            width: 100%;
            height: 135px;
            object-fit: cover;
            border-radius: 8px;
            box-shadow: 0 3px 15px rgba(0,0,0,0.12);
            transition: all 0.3s ease;
        }

        .desktop-layout .article-card-img:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 25px rgba(0,0,0,0.18);
        }

        .desktop-layout .article-header-content {
            flex: 1;
            padding-top: 10px;
        }

        .desktop-layout .article-spacing {
            height: 10px;
        }

        .desktop-layout .article-body {
            margin-top: 10px;
        }

        .desktop-layout .author-info {
            margin-top: 5px;
        }

        /* Stiluri pentru MOBIL */
        .mobile-layout {
            text-align: left;
        }

        .mobile-image-container {
            width: 100%;
            margin-bottom: 15px;
        }

        .mobile-article-img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 8px;
            box-shadow: 0 3px 15px rgba(0,0,0,0.12);
        }

        .mobile-title {
            margin-bottom: 15px;
        }

        .mobile-lead {
            margin-bottom: 15px;
        }

        .mobile-date {
            margin-bottom: 20px;
        }
    </style>'''

def extract_article_data(article):
    """Extrage datele din vechiul format de articol"""
    try:
        # Extrage titlul
        title_link = article.find('h2', class_='custom-h1').find('a')
        title = title_link.get_text(strip=True)
        url = title_link.get('href')

        # Extrage data
        time_elem = article.find('time')
        date = time_elem.get_text(strip=True).replace('On ', '')

        # Extrage categoria
        category_link = article.find('a', class_='color-green')
        category = category_link.get_text(strip=True)
        category_url = category_link.get('href')

        # Extrage autorul
        author_elem = article.find('span', id='hidden2')
        if author_elem:
            author = author_elem.get_text(strip=True).replace('by ', '')
        else:
            author = "Neculai Fantanaru"

        # Extrage descrierea
        desc_p = article.find('p', class_='mb-35px')
        description = desc_p.get_text(strip=True) if desc_p else ""

        return title, url, date, category, category_url, author, description

    except Exception as e:
        print(f"Eroare la extragerea datelor din articol: {e}")
        return None, None, None, None, None, None, None

def process_html_file(file_path, base_url="https://neculaifantanaru.com"):
    """Procesează un fișier HTML și îl actualizează cu noul format"""

    print(f"Procesez: {file_path}")

    try:
        # Citește fișierul
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        soup = BeautifulSoup(content, 'html.parser')

        # Adaugă CSS-ul în head dacă nu există deja
        head = soup.find('head')
        if head and not head.find('style', string=re.compile('article-card-new')):
            css_style = BeautifulSoup(get_article_css(), 'html.parser')
            head.append(css_style)

        # Găsește comentariile START și FINAL
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        start_comment = None
        end_comment = None

        for comment in comments:
            if 'ARTICOL START' in comment:
                start_comment = comment
            elif 'ARTICOL FINAL' in comment:
                end_comment = comment
                break

        if not start_comment or not end_comment:
            print(f"Nu am găsit comentariile în {file_path}")
            return

        # Găsește toate articolele între comentarii
        current = start_comment.next_sibling
        articles_to_process = []

        while current and current != end_comment:
            if hasattr(current, 'name') and current.name == 'article':
                if 'blog-box' in current.get('class', []):
                    articles_to_process.append(current)
            current = current.next_sibling

        print(f"Găsite {len(articles_to_process)} articole în {file_path}")

        # Procesează fiecare articol
        new_articles_html = []

        for i, article in enumerate(articles_to_process):
            print(f"  Procesez articolul {i+1}/{len(articles_to_process)}")

            # Extrage datele
            title, url, date, category, category_url, author, description = extract_article_data(article)

            if not title:
                continue

            # Construiește URL-ul complet dacă este relativ
            if url.startswith('/'):
                url = base_url + url
            elif not url.startswith('http'):
                url = base_url + '/' + url

            # Extrage imaginea din pagina articolului
            img_src, img_alt = get_image_from_article_page(url)

            if not img_src:
                print(f"    Nu am găsit imagine pentru {title}")
                # Folosește o imagine default sau skip
                img_src = "https://neculaifantanaru.com/images/default-article.jpg"
                img_alt = title
            else:
                print(f"    Găsită imagine: {img_src}")

            # Construiește URL-ul complet pentru imagine dacă este relativ
            if img_src.startswith('/'):
                img_src = base_url + img_src
            elif not img_src.startswith('http'):
                img_src = base_url + '/' + img_src

            # Creează noul HTML
            new_article_html = create_new_article_html(
                title, url, date, category, category_url, author, description, img_src, img_alt
            )

            new_articles_html.append(new_article_html)

            # Pauză scurtă pentru a nu suprasolicita serverul
            time.sleep(0.5)

        # Înlocuiește vechile articole cu cele noi PĂSTRÂND ORDINEA
        if new_articles_html:
            # Șterge vechile articole
            for article in articles_to_process:
                article.decompose()

            # Creează toate articolele noi într-un singur string
            combined_html = '\n\t\t\t\t\t'.join(new_articles_html)
            new_soup = BeautifulSoup(combined_html, 'html.parser')

            # Inserează toate articolele noi după comentariul START, în ordinea corectă
            insert_position = start_comment
            for new_article in new_soup.find_all('article'):
                insert_position.insert_after(new_article)
                insert_position = new_article  # Următorul articol se inserează după cel curent

        # PARTEA CORECTATĂ: Gestionarea backup-ului
        backup_path = file_path + '.backup'

        # Șterge backup-ul vechi dacă există
        if os.path.exists(backup_path):
            os.remove(backup_path)
            print(f"  Șters backup vechi: {backup_path}")

        # Creează backup
        if os.path.exists(file_path):
            os.rename(file_path, backup_path)
            print(f"  Creat backup: {backup_path}")

        # Salvează fișierul modificat
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(soup))

        print(f"✓ Finalizat și salvat: {file_path}")

    except Exception as e:
        print(f"✗ Eroare la procesarea {file_path}: {e}")

def main():
    """Funcția principală"""

    # Lista fișierelor
    files = [
        "lideri-si-atitudine.html", "leadership-magic.html", "leadership-de-succes.html",
        "hr-resurse-umane.html", "legile-conducerii.html", "leadership-total.html",
        "leadership-de-durata.html", "principiile-conducerii.html", "leadership-plus.html",
        "calitatile-unui-lider.html", "leadership-de-varf.html", "leadership-impact.html",
        "dezvoltare-personala.html", "aptitudini-si-abilitati-de-leadership.html",
        "leadership-real.html", "leadership-de-baza.html", "leadership-360.html",
        "leadership-pro.html", "leadership-expert.html", "leadership-know-how.html",
        "jurnal-de-leadership.html", "alpha-leadership.html", "leadership-on-off.html",
        "leadership-deluxe.html", "leadership-xxl.html", "leadership-50-extra.html",
        "leadership-fusion.html", "leadership-v8.html", "leadership-x3-silver.html",
        "leadership-q2-sensitive.html", "leadership-t7-hybrid.html", "leadership-n6-celsius.html",
        "leadership-s4-quartz.html", "leadership-gt-accent.html", "leadership-fx-intensive.html",
        "leadership-iq-light.html", "leadership-7th-edition.html", "leadership-xs-analytics.html",
        "leadership-z3-extended.html", "leadership-ex-elite.html", "leadership-w3-integra.html",
        "leadership-sx-experience.html", "leadership-y5-superzoom.html", "performance-ex-flash.html",
        "leadership-mindware.html", "leadership-r2-premiere.html", "leadership-y4-titanium.html",
        "leadership-quantum-xx.html"
    ]

    base_path = r"e:\Carte\BB\17 - Site Leadership\Principal 2022\ro"

    print(f"Încep procesarea a {len(files)} fișiere...")
    print("=" * 50)

    for file_name in files:
        file_path = os.path.join(base_path, file_name)

        if os.path.exists(file_path):
            process_html_file(file_path)
        else:
            print(f"✗ Fișierul nu există: {file_path}")

        print("-" * 30)

    print("=" * 50)
    print("Finalizat!")

if __name__ == "__main__":
    main()