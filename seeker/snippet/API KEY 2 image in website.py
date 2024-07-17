#date: 2024-07-17T17:08:44Z
#url: https://api.github.com/gists/26176d2ec38dc76acf34704a46bef75a
#owner: https://api.github.com/users/me-suzy

import os
import re
from openai import OpenAI
import requests
from PIL import Image
import io

# Configurare OpenAI API
client = OpenAI(api_key='YOUR-KEY')

def generate_image_description(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Ești un expert în crearea de descrieri concise pentru imagini. Analizează textul dat și creează o descriere scurtă și clară pentru o imagine care să reprezinte esența întregului mesaj."},
                {"role": "user", "content": f"Creează o descriere scurtă pentru o imagine care să reprezinte esența următorului text, în maxim 50 de cuvinte:\n\n{text}"}
            ]
        )
        description = response.choices[0].message.content.strip()
        return description
    except Exception as e:
        print(f"Eroare la generarea descrierii imaginii: {e}")
        return "Descriere temporar indisponibilă"

def generate_image(description):
    try:
        enhanced_prompt = f"""
        Creează o imagine bazată pe următoarea descriere, fără a include text sau fraze scrise în imagine:

        {description}

        Instrucțiuni suplimentare:
        - Nu include niciun fel de text, cuvinte sau litere în imagine.
        - Concentrează-te pe reprezentarea vizuală a conceptelor și ideilor.
        - Utilizează simboluri, forme și culori pentru a transmite mesajul.
        - Asigură-te că imaginea este clară și ușor de înțeles fără text explicativ.
        """

        response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        print(f"Eroare la generarea imaginii: {e}")
        return None

def generate_image_caption(description):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generează un scurt caption pentru o imagine bazată pe următoarea descriere. Caption-ul trebuie să fie concis și să reflecte conceptul de leadership sau performanță."},
                {"role": "user", "content": description}
            ]
        )
        caption = response.choices[0].message.content.strip()
        return caption
    except Exception as e:
        print(f"Eroare la generarea caption-ului imaginii: {e}")
        return "Concept de performanță în leadership"

def download_and_resize_image(url, output_path):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    img = img.resize((1024, 719), Image.LANCZOS)
    img.save(output_path)
    return output_path

def extract_article_content(content):
    start_tag = '<!-- ARTICOL START -->'
    end_tag = '<!-- ARTICOL FINAL -->'
    pattern = re.compile(f'{re.escape(start_tag)}(.*?){re.escape(end_tag)}', re.DOTALL)
    match = pattern.search(content)
    if match:
        return match.group(1)
    return None

def clean_html(html_content):
    clean_text = re.sub(r'<[^>]+>', '', html_content)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def process_file(file_path, image_folder):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    article_content = extract_article_content(content)
    if not article_content:
        print(f"Nu s-a găsit conținut de articol în {file_path}")
        return False

    cleaned_content = clean_html(article_content)
    description = generate_image_description(cleaned_content)
    image_url = generate_image(description)
    
    if image_url:
        file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_image.jpg"
        image_path = os.path.join(image_folder, file_name)
        local_image_path = download_and_resize_image(image_url, image_path)
        
        caption = generate_image_caption(description)
        
        image_html = f'''
        <br><br>
        <div class="feature-img-wrap">
            <img src="https://neculaifantanaru.com/images/{file_name}" alt="{description[:100]}" class="img-responsive">
            <!-- <p class="image-caption">Ilustrație: {caption}</p> -->
        </div>
        '''
        
        content = content.replace('YYY', image_html, 1)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

        return True
    
    return False

def main():
    folder_path = 'd:\\1\\'
    image_folder = 'd:\\1\\images\\'
    os.makedirs(image_folder, exist_ok=True)
    modified_files = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.html'):
            file_path = os.path.join(folder_path, filename)
            was_modified = process_file(file_path, image_folder)
            if was_modified:
                modified_files.append(file_path)

    with open('api+rezumat.txt', 'w', encoding='utf-8') as file:
        file.write("Fișiere modificate:\n")
        for modified_file in modified_files:
            file.write(f"{modified_file}\n")

if __name__ == "__main__":
    main()
