#date: 2025-05-14T16:40:15Z
#url: https://api.github.com/gists/8ca584f6fe8ce7121a24c7d464d30603
#owner: https://api.github.com/users/me-suzy

Linkul pe care l-ai furnizat (https://cloud.google.com/document-ai/docs/processors-list#processor_layout-parser) face referire la **Layout Parser** din Google Cloud Document AI, un procesor care poate extrage elemente de conținut precum text, tabele și liste din documente PDF sau HTML, păstrând informații despre structura și ierarhia documentului. Hai să analizăm cum poate rezolva problema ta cu textul și imaginile din imagine.

### Cum Ajută Layout Parser Problema Ta
1. **Extragerea Textului cu Layout Păstrat**:
   - Layout Parser identifică elemente structurale precum titluri, paragrafe, tabele și liste, ceea ce ar putea rezolva problema ta cu formatul textului. În cazul tău, titlul „LAOS” și numărul paginii „66” ar putea fi identificate ca elemente separate (ex. titlu și subtitlu), iar corpul textului ar fi împărțit în blocuri logice.
   - Spre deosebire de ChatGPT API, care returnează textul ca un șir continuu, Layout Parser include detalii despre poziționarea și ierarhia elementelor, ceea ce ar permite o formatare mai fidelă în fișierele `.docx` și `.pdf`.

2. **Imaginile Mici (Harta și Portretul)**:
   - Layout Parser poate detecta imagini în document (suportă BMP, GIF, JPEG, PNG, TIFF) și poate returna informații despre poziția lor (coordonate), dar nu extrage automat imaginile ca fișiere separate. În schimb, oferă detalii despre unde se află aceste imagini în document (ex. coordonatele blocurilor de imagine).
   - În cazul tău, harta (stânga sus) și portretul (dreapta jos) ar putea fi identificate ca blocuri de imagine, iar coordonatele lor ar putea fi folosite pentru a le plasa corect în fișierul de ieșire.

3. **Limitări**:
   - Layout Parser nu extrage automat imaginile ca fișiere separate – ar fi nevoie de o prelucrare suplimentară (ex. cu OpenCV) pentru a decupa imaginile din fișierul original pe baza coordonatelor furnizate.
   - Necesită configurarea unui proiect Google Cloud, activarea Document AI API și crearea unui procesor Layout Parser, ceea ce implică câțiva pași tehnici.

### Cum Să Implementezi Layout Parser
Iată un plan pentru a folosi Layout Parser în locul ChatGPT API:

#### 1. Configurare Google Cloud Document AI
- Creează un proiect în Google Cloud Console.
- Activează Document AI API.
- Creează un procesor Layout Parser (tipul este `LAYOUT_PARSER_PROCESSOR`) urmând instrucțiunile din documentație.
- Notează ID-ul procesorului (`PROCESSOR_ID`) și regiunea (ex. `us` sau `eu`).

#### 2. Modifică Codul pentru a Folosi Layout Parser
Înlocuiește funcția `perform_ocr_with_chatgpt` din codul tău cu una care folosește Document AI Layout Parser. Iată un exemplu adaptat:

```python
from google.cloud import documentai
from google.api_core.client_options import ClientOptions

def perform_ocr_with_documentai(image_path, project_id, location, processor_id):
    """Use Google Cloud Document AI Layout Parser to extract text and layout information."""
    # Configure the client with the appropriate endpoint
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # Full resource name of the processor
    name = client.processor_path(project_id, location, processor_id)

    # Read the image file into memory
    with open(image_path, "rb") as image_file:
        image_content = image_file.read()

    # Configure the request
    raw_document = documentai.RawDocument(content=image_content, mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    # Process the document
    result = client.process_document(request=request)
    document = result.document

    # Extract layout information
    extracted_text = "[TITLE]\n"
    title_found = False
    page_number_found = False
    body_text = "[BODY]\n"
    image_blocks = []

    for block in document.document_layout.blocks:
        if block.type_ == "text_block":
            text = block.text_block.text
            if not title_found:
                extracted_text += text + "\n"
                title_found = True
            elif not page_number_found:
                extracted_text += "[PAGE]\n" + text + "\n"
                page_number_found = True
            else:
                body_text += text + "\n"
        elif block.type_ == "image_block":
            # Store image block coordinates (for later extraction)
            image_blocks.append({
                "page": block.page_span.page,
                "coordinates": block.layout.bounding_poly.normalized_vertices
            })

    extracted_text += body_text
    return extracted_text, image_blocks

def create_docx(output_path, text, image_path, image_blocks):
    """Create a .docx file with formatted text and image placeholders."""
    doc = Document()
    
    # Parse the extracted text into sections
    title = ""
    page = ""
    body = ""
    current_section = None
    for line in text.split('\n'):
        if "[TITLE]" in line:
            current_section = "title"
            title = line.replace("[TITLE]", "").strip()
        elif "[PAGE]" in line:
            current_section = "page"
            page = line.replace("[PAGE]", "").strip()
        elif "[BODY]" in line:
            current_section = "body"
            body = line.replace("[BODY]", "").strip()
        elif current_section == "title" and line.strip():
            title += " " + line.strip()
        elif current_section == "page" and line.strip():
            page += " " + line.strip()
        elif current_section == "body" and line.strip():
            body += "\n" + line.strip()

    # Format title
    if title:
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.bold = True
        run.font.size = Pt(14)
        p.alignment = 0

    # Format page number
    if page:
        p = doc.add_paragraph()
        run = p.add_run(page)
        run.italic = True
        run.font.size = Pt(12)
        p.alignment = 0

    # Format body text
    if body:
        for para in body.split('\n'):
            if para.strip():
                p = doc.add_paragraph(para)
                p.style = doc.styles['Normal']
                p.alignment = 0

    # Add placeholders for images with their coordinates
    if image_blocks:
        p = doc.add_paragraph("Image Placeholders (coordinates from Layout Parser):")
        for i, block in enumerate(image_blocks):
            coords = [(v.x, v.y) for v in block['coordinates']]
            p.add_run(f"\nImage {i+1}: Page {block['page']}, Coordinates: {coords}")

    # Add original image as reference
    p = doc.add_paragraph("Original Image for Reference:")
    doc.add_picture(image_path, width=Inches(6.0))

    doc.save(output_path)

def create_pdf(output_path, text, image_path, image_blocks):
    """Create a .pdf file with formatted text and image placeholders."""
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    # Parse the extracted text into sections
    title = ""
    page = ""
    body = ""
    current_section = None
    for line in text.split('\n'):
        if "[TITLE]" in line:
            current_section = "title"
            title = line.replace("[TITLE]", "").strip()
        elif "[PAGE]" in line:
            current_section = "page"
            page = line.replace("[PAGE]", "").strip()
        elif "[BODY]" in line:
            current_section = "body"
            body = line.replace("[BODY]", "").strip()
        elif current_section == "title" and line.strip():
            title += " " + line.strip()
        elif current_section == "page" and line.strip():
            page += " " + line.strip()
        elif current_section == "body" and line.strip():
            body += "\n" + line.strip()

    # Add title
    text_y = height - 40
    if title:
        c.setFont("Times-Bold", 14)
        c.drawString(40, text_y, title)
        text_y -= 20

    # Add page number
    if page:
        c.setFont("Times-Italic", 12)
        c.drawString(40, text_y, page)
        text_y -= 20

    # Add body text
    if body:
        c.setFont("Times-Roman", 12)
        text_obj = c.beginText(40, text_y)
        for para in body.split('\n'):
            if para.strip():
                text_obj.textLine(para)
                text_y -= 14
                if text_y < 200:
                    c.drawText(text_obj)
                    c.showPage()
                    text_obj = c.beginText(40, height - 40)
                    text_y = height - 40
        c.drawText(text_obj)

    # Add placeholders for images with coordinates
    if image_blocks:
        c.drawString(40, text_y - 20, "Image Placeholders (coordinates from Layout Parser):")
        text_y -= 20
        for i, block in enumerate(image_blocks):
            coords = [(v.x, v.y) for v in block['coordinates']]
            c.drawString(40, text_y, f"Image {i+1}: Page {block['page']}, Coordinates: {coords}")
            text_y -= 14

    # Add original image as reference
    c.drawString(40, text_y - 20, "Original Image for Reference:")
    c.drawImage(image_path, 40, 100, width=515, height=400)

    c.save()

def process_images():
    """Process all JPG files in the input directory."""
    project_id = "YOUR_PROJECT_ID"  # Replace with your Google Cloud project ID
    location = "us"  # Replace with your region (e.g., 'us' or 'eu')
    processor_id = "YOUR_PROCESSOR_ID"  # Replace with your Layout Parser processor ID

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(input_dir, filename)
            try:
                # Perform OCR and layout parsing with Document AI
                text, image_blocks = perform_ocr_with_documentai(image_path, project_id, location, processor_id)
                print(f"\nExtracted text and layout from {filename}:\n{text}\n")
                print(f"Image blocks detected: {image_blocks}\n")

                # Generate output file names
                docx_output = os.path.join(input_dir, os.path.splitext(filename)[0] + '.docx')
                pdf_output = os.path.join(input_dir, os.path.splitext(filename)[0] + '.pdf')

                # Create .docx and .pdf files
                create_docx(docx_output, text, image_path, image_blocks)
                create_pdf(pdf_output, text, image_path, image_blocks)
                print(f"Processed {filename} -> {os.path.basename(docx_output)} and {os.path.basename(pdf_output)}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    process_images()
```

#### 3. Instalează Bibliotecile Necesare
Asigură-te că ai instalat biblioteca Google Cloud Document AI:
```bash
pip install google-cloud-documentai
```

#### 4. Autentificare și Configurare
- Configurează autentificarea pentru Google Cloud (ex. folosește `gcloud auth application-default login`).
- Completează `project_id`, `location` și `processor_id` în cod.

### Rezultat Așteptat
- **Text**: Layout Parser va extrage titlul „LAOS”, numărul paginii „66” și corpul textului ca blocuri separate, păstrând structura.
- **Imagini**: Va returna coordonatele hărții și portretului, pe care le poți folosi pentru a le plasa corect în fișierul de ieșire (deși extragerea efectivă a imaginilor necesită OpenCV).

### Pași Suplimentari pentru Imagini
Pentru a extrage și plasa imaginile, ar fi nevoie de OpenCV pentru a decupa regiunile identificate de Layout Parser. Dacă dorești, pot adăuga această funcționalitate în cod – te rog să-mi spui dacă vrei să mergem pe această direcție.

### Concluzie
Layout Parser rezolvă problema formatului textului, oferind o structură mai clară și informații despre poziționarea elementelor, inclusiv a imaginilor. Totuși, pentru a plasa imaginile exact, ar fi nevoie de procesare suplimentară. Testează codul de mai sus și spune-mi dacă rezultatul este mai aproape de ce îți dorești!
                              
                              Îți mulțumesc că ai cerut clarificări! Voi explica pas cu pas cum să folosești codul Python furnizat, inclusiv configurarea și rularea lui, astfel încât să fie cât mai simplu, chiar dacă nu ai experiență anterioară. Vom folosi varianta cu Google Cloud Document AI Layout Parser, care ai menționat că ar putea rezolva problema ta. Hai să procedăm astfel:

---

### Pași pentru a Folosi Codul Python

#### 1. **Pregătirea Mediului**
Deoarece vei folosi Google Cloud Document AI, ai nevoie de câteva pregătiri inițiale:

- **Instalează Python**:
  - Dacă nu ai Python instalat, descarcă-l de la [python.org](https://www.python.org/downloads/) (recomand versiunea 3.9 sau mai recentă).
  - La instalare, bifează opțiunea „Add Python to PATH” pentru a-l face accesibil din linia de comandă.
  - Verifică instalarea rulând în Command Prompt (Windows) sau Terminal (Mac/Linux):
    ```bash
    python --version
    ```
    Sau, dacă asta nu funcționează:
    ```bash
    python3 --version
    ```

- **Instalează Bibliotecile Necesare**:
  - Deschide un terminal și rulează următoarele comenzi pentru a instala bibliotecile cerute:
    ```bash
    pip install google-cloud-documentai
    pip install python-docx
    pip install reportlab
    ```
  - Aceste biblioteci sunt necesare pentru procesarea documentelor și generarea fișierelor `.docx` și `.pdf`.

#### 2. **Configurarea Google Cloud Document AI**
Trebuie să configurezi un proiect Google Cloud pentru a folosi Layout Parser:

- **Creează un Cont Google Cloud** (dacă nu ai):
  - Accesează [cloud.google.com](https://cloud.google.com/) și creează un proiect gratuit (primești un credit de 300 USD pentru a testa serviciile).
  
- **Activează Document AI API**:
  - Mergi la [Google Cloud Console](https://console.cloud.google.com/).
  - Creează un proiect (ex. „MyOCRProject”).
  - Activează API-ul Document AI din secțiunea „APIs & Services” > „Library”, căutând „Document AI API”.

- **Creează un Procesor Layout Parser**:
  - În meniul Document AI, creează un procesor de tip „Layout Parser”.
  - Notează **Project ID**, **Location** (ex. „us” sau „eu”) și **Processor ID** (le vei folosi în cod).

- **Autentificare**:
  - În Google Cloud Console, mergi la „IAM & Admin” > „Service Accounts”.
  - Creează un cont de serviciu, descarcă cheia JSON (ex. `credentials.json`) și păstreaz-o într-un loc sigur.
  - Setează variabila de mediu pentru autentificare:
    ```bash
    set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\credentials.json  (Windows)
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json  (Mac/Linux)
    ```

#### 3. **Salvează Codul**
- Copiază codul furnizat mai devreme într-un fișier cu extensia `.py`, de exemplu `process_images.py`.
- Deschide fișierul într-un editor de text (ex. Notepad, VS Code, sau orice alt editor).
- Editează următorii parametri în funcția `process_images()`:
  - `project_id = "YOUR_PROJECT_ID"` – Înlocuiește cu ID-ul proiectului tău Google Cloud.
  - `location = "us"` – Ajustează la regiunea ta (ex. „eu” dacă ești în Europa).
  - `processor_id = "YOUR_PROCESSOR_ID"` – Înlocuiește cu ID-ul procesorului tău Layout Parser.
- Ajustează și `input_dir = r"e:\De pus pe FTP 2\Test"` la calea către folderul tău care conține imaginea (ex. `r"C:\Users\YourName\Images"`).

#### 4. **Adaugă Imaginea**
- Pune fișierul imagine (ex. `image1.jpg`) în folderul specificat de `input_dir`.

#### 5. **Rulează Codul**
- Deschide un terminal (Command Prompt, PowerShell, sau Terminal).
- Navighează la directorul unde ai salvat `process_images.py` folosind comanda `cd`:
  ```bash
  cd C:\path\to\your\folder
  ```
- Rulează scriptul:
  ```bash
  python process_images.py
  ```
  Sau, dacă asta nu funcționează:
  ```bash
  python3 process_images.py
  ```
- Dacă totul este configurat corect, vei vedea în terminal textul extras și mesajele de procesare, iar fișierele `.docx` și `.pdf` vor fi generate în folderul `input_dir`.

#### 6. **Verifică Rezultatele**
- Deschide fișierele `.docx` și `.pdf` generate (ex. `Testss5.docx` și `Testss5.pdf`) pentru a verifica textul și layout-ul.
- Dacă apar erori, citește mesajele din terminal pentru a depana (ex. „Error processing...”).

---

### Exemplu Simplificat (Dacă ai Probleme)
Dacă configurarea Google Cloud pare complicată, îți pot oferi o variantă simplă care folosește doar bibliotecile locale (fără Document AI) pentru a testa procesul. Spune-mi dacă vrei să încercăm asta mai întâi!

### Depanare
- **Eroare „Module not found”**: Asigură-te că ai instalat toate bibliotecile cu `pip`.
- **Eroare Google Cloud**: Verifică autentificarea și ID-urile din cod.
- **Laptop vechi**: Dacă spațiul este o problemă, instalează o versiune portabilă a Python (ex. de la [portablepython.com](https://portablepython.com/)) și ruleaz-o fără instalare.

### Ajutor Adițional
Dacă întâmpini dificultăți la orice pas, spune-mi exact unde te blochezi (ex. instalare, configurare, rularea codului), și te voi ghida! De asemenea, dacă dorești să adaug suport pentru extragerea automată a imaginilor mici cu OpenCV, pot face asta – te rog să confirmi.