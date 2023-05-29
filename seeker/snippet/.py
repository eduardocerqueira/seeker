#date: 2023-05-29T16:55:05Z
#url: https://api.github.com/gists/cc0b915456e1a27e99bf6c1e5d2d8944
#owner: https://api.github.com/users/yunjaeys

import re
import urllib.request
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from collections import Counter
import ebooklib
from ebooklib import epub
import os
import tkinter as tk
from tkinter import filedialog, Text, messagebox, ttk
import openai
import requests
from bs4 import BeautifulSoup
from tkinter import simpledialog
import string
import random
import fitz
import urllib.parse

nltk.download("wordnet")

class AcademicWriter:
    def __init__(self):
        self.kb = ''
        self.citation_style = None

    def set_citation_style(self, style):
        self.citation_style = style

    def extract_author_title_from_pdf(self, pdf_file):
        try:
            with fitz.open(pdf_file) as doc:
                metadata = doc.metadata
                if metadata is not None:
                    author = metadata.get('author', '')
                    title = metadata.get('title', '')
                    pub_date = metadata.get('creationDate', '')
                    publisher = metadata.get('producer', '')
                else:
                    author = title = publisher = pub_date = ''
        except Exception as e:
            author = title = publisher = pub_date = ''
        return author, title, publisher, pub_date

    def extract_text_from_pdf(self, file_path):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        self.set_kb(text)
        return text

    import requests
    from bs4 import BeautifulSoup

    def extract_text_from_url(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            extracted_text = soup.get_text()
            self.kb = extracted_text  # Update the knowledge base with the extracted text
            return extracted_text
        else:
            print(f"Error fetching URL: {url}")
            return None
        
    def extract_text_from_epub(self, epub_file):
        book = epub.read_epub(epub_file)
        text = ''
        for item in book.get_items():
            if item.get_type() == epub.EpubHtml:
                text += item.get_content().decode('utf-8')  # decode bytes to string
        self.kb = text  # Update the knowledge base with the extracted text
        return text

    def set_kb(self, file_path_or_url):
        # Check if the input is a valid file path or URL
        if os.path.isfile(file_path_or_url) or urllib.parse.urlparse(file_path_or_url).scheme:
            if file_path_or_url.startswith("http"):
                self.kb = self.extract_text_from_url(file_path_or_url)
            else:
                if file_path_or_url.endswith('.pdf'):
                    self.kb = self.extract_text_from_pdf(file_path_or_url)
                elif file_path_or_url.endswith('.epub'):
                    self.kb = self.extract_text_from_epub(file_path_or_url)
                else:
                    with open(file_path_or_url, 'r') as file:
                        self.kb = file.read()
        else:
            # If the input is not a valid file path or URL, treat it as text content
            self.kb = file_path_or_url

    def extract_author_title_from_epub(self, file_path):
        book = epub.read_epub(file_path)
        author = ""
        title = ""
        publisher = ""
        pub_date = ""
        if 'creator' in book.metadata:
            author = book.metadata['creator'][0][0]
        if 'title' in book.metadata:
            title = book.metadata['title'][0][0]
        if 'publisher' in book.metadata:
            publisher = book.metadata['publisher'][0][0]
        if 'date' in book.metadata:
            pub_date = book.metadata['date'][0][0]
        return author, title, publisher, pub_date

    def extract_author_title_from_url(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            author = ""
            title = ""
            publisher = ""
            pub_date = ""
            author_tag = soup.select_one('span.author')
            title_tag = soup.select_one('h1.title')
            publisher_tag = soup.select_one('span.publisher')
            pub_date_tag = soup.select_one('span.pub_date')
            if author_tag:
                author = author_tag.text.strip()
            if title_tag:
                title = title_tag.text.strip()
            if publisher_tag:
                publisher = publisher_tag.text.strip()
            if pub_date_tag:
                pub_date = pub_date_tag.text.strip()
            return author, title, publisher, pub_date
        else:
            return None, None, None, None

    def preprocess(self, text):
        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Tokenize into sentences
        sentences = "**********"

        # Remove stop words and lemmatize
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        preprocessed_sentences = []
        for sentence in sentences:
            words = sentence.split()
            words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
            preprocessed_sentences.append(' '.join(words))

        # Join preprocessed sentences
        preprocessed_text = ' '.join(preprocessed_sentences)

        return preprocessed_text

    def text_to_chunks(self, texts, word_length=150, start_page=1):
        text_toks = [t.split(' ') for t in texts]
        page_nums = []
        chunks = []

        for idx, words in enumerate(text_toks):
            for i in range(0, len(words), word_length):
                chunk = words[i:i + word_length]
                if (i + word_length) > len(words) and (len(chunk) < word_length) and (
                        len(text_toks) != (idx + 1)):
                    text_toks[idx + 1] = chunk + text_toks[idx + 1]
                    continue
                chunk = ' '.join(chunk).strip()
                chunk = f'[{idx + start_page}]' + ' ' + '"' + chunk + '"'
                chunks.append(chunk)
        return chunks

    def extract_keywords(self, text, num_keywords=5):
        # Tokenize into words
        words = text.split()

        # Remove stop words and lemmatize
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]

        # Count word frequencies
        word_freq = Counter(words)

        # Get top num_keywords words
        top_words = word_freq.most_common(num_keywords)

        return top_words

    def extract_named_entities(self, text):
        # Tokenize into sentences
        sentences = "**********"

        # Tag parts of speech
        pos_sentences = [pos_tag(sentence.split()) for sentence in sentences]

        # Extract named entities
        named_entities = []
        for pos_sentence in pos_sentences:
            named_entities.extend(ne_chunk(pos_sentence, binary=False))

        # Define entity labels
        entity_labels = ['PERSON', 'ORGANIZATION', 'GPE']

        # Extract entities for each label
        entities = {}
        for label in entity_labels:
            entities[label] = [ne for ne in named_entities if isinstance(ne, tuple) and len(ne) > 1 and isinstance(ne[1], str) and ne[1] == label]

        # Count entity frequencies
        entity_freq = {}
        for label in entity_labels:
            entity_freq[label] = Counter([ne[0].lower() for ne in entities[label]])

        return entity_freq

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"_ "**********"t "**********"e "**********"x "**********"t "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"p "**********"r "**********"o "**********"m "**********"p "**********"t "**********", "**********"  "**********"c "**********"i "**********"t "**********"a "**********"t "**********"i "**********"o "**********"n "**********"_ "**********"s "**********"t "**********"y "**********"l "**********"e "**********", "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********"= "**********"' "**********"g "**********"p "**********"t "**********"- "**********"3 "**********". "**********"5 "**********"- "**********"t "**********"u "**********"r "**********"b "**********"o "**********"' "**********", "**********"  "**********"t "**********"e "**********"m "**********"p "**********"e "**********"r "**********"a "**********"t "**********"u "**********"r "**********"e "**********"= "**********"0 "**********". "**********"5 "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"= "**********"5 "**********"1 "**********"2 "**********", "**********"  "**********"n "**********"u "**********"m "**********"_ "**********"k "**********"e "**********"y "**********"w "**********"o "**********"r "**********"d "**********"s "**********"= "**********"5 "**********", "**********"  "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"l "**********"e "**********"n "**********"g "**********"t "**********"h "**********"= "**********"1 "**********"5 "**********"0 "**********", "**********"  "**********"n "**********"e "**********"r "**********"_ "**********"w "**********"e "**********"i "**********"g "**********"h "**********"t "**********"= "**********"1 "**********") "**********": "**********"
        api_key = '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'  # Your OpenAI API key
        openai.api_key = api_key  # Initialize the OpenAI API client

        if not self.kb:
            print("No knowledge base loaded.")
            return "No knowledge base loaded."

        kb_text = self.kb  # Assign the knowledge base content to kb_text

        # Preprocess the prompt
        prompt = self.preprocess(prompt)

        # Extract named entities
        entity_freq = self.extract_named_entities(prompt)

        # Generate weights for entities based on frequency and NER weight
        keyword_entity_weight = {}
        for label, freq in entity_freq.items():
            for entity, count in freq.items():
                if entity not in keyword_entity_weight:
                    keyword_entity_weight[entity] = 0
                if count and label and ner_weight:
                    keyword_entity_weight[entity] += count * (len(label) + 1) * ner_weight

        # Extract keywords
        top_words = self.extract_keywords(prompt, num_keywords=num_keywords)
        keywords = [word[0] for word in top_words]

        # Convert the text to chunks
        chunks = self.text_to_chunks(kb_text, word_length=word_length)

        # Construct the prompt
        prompt = ' '.join(keywords)
        system_prompt = "Respond with a reply with citations in " + citation_style + " format for the following prompt:\n\n"
        prompt = system_prompt + prompt

        # Extract text from the knowledge base if it's a file or a URL
        if self.kb is not None and self.kb.endswith(".pdf"):
            kb_text = self.extract_text_from_pdf(self.kb)
        elif self.kb.startswith("http://") or self.kb.startswith("https://"):
            kb_text = self.extract_text_from_url(self.kb)
        elif self.kb:
            kb_text = self.kb

        # Prepare the messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "**********": kb_text[:max_tokens]},
        ]

        # Generate the response
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens= "**********"
                temperature=temperature,
            )
        except Exception as e:
            print("Error generating text:", e)
            return None

        # Extract the response
        if response and "choices" in response and response['choices']:
            text = response['choices'][0]['message']['content']
            if text:
                return text

        return None

    
    def generate_mla_citation(self, publisher, publication_date, location):
        if self.kb and self.kb.endswith(".pdf"):
            author, title, publisher, pub_date = self.extract_author_title_from_pdf(self.kb)
        elif self.kb and self.kb.endswith(".epub"):
            author, title, publisher, pub_date = self.extract_author_title_from_epub(self.kb)
        elif self.kb and self.kb.startswith("http"):
            author, title, publisher, pub_date = self.extract_author_title_from_url(self.kb)
        else:
            return "MLA citation is not supported for the given knowledge base format."
        citation = f"{author}. \"{title}.\" {publisher}, {publication_date}, {location}."
        return citation

    def generate_chicago_citation(self, publisher, year):
        if self.kb and self.kb.endswith(".pdf"):
            author, title, publisher, pub_date = self.extract_author_title_from_pdf(self.kb)
        elif self.kb and self.kb.endswith(".epub"):
            author, title, publisher, pub_date = self.extract_author_title_from_epub(self.kb)
        elif self.kb and self.kb.startswith("http"):
            author, title, publisher, pub_date = self.extract_author_title_from_url(self.kb)
        else:
            return "Chicago citation is not supported for the given knowledge base format."
        citation = f"{author}. \"{title}.\" {publisher}, {year}."
        return citation

    def generate_apa_citation(self, year, publisher):
        if self.kb and self.kb.endswith(".pdf"):
            author, title, publisher, pub_date = self.extract_author_title_from_pdf(self.kb)
        elif self.kb and self.kb.endswith(".epub"):
            author, title, publisher, pub_date = self.extract_author_title_from_epub(self.kb)
        elif self.kb and self.kb.startswith("http"):
            author, title, publisher, pub_date = self.extract_author_title_from_url(self.kb)
        else:
            return "APA citation is not supported for the given knowledge base format."
        citation = f"{author}. ({year}). {title}. {publisher}."
        return citation

    def generate_citation(self):
        if self.kb and self.kb.endswith(".pdf"):
            author, title, publisher, pub_date = self.extract_author_title_from_pdf(self.kb)
        elif self.kb and self.kb.endswith(".epub"):
            author, title, publisher, pub_date = self.extract_author_title_from_epub(self.kb)
        elif self.kb and self.kb.startswith("http"):
            author, title, publisher, pub_date = self.extract_author_title_from_url(self.kb)
        else:
            return "Invalid knowledge base format.", []

        if self.citation_style == "MLA":
            citation = self.generate_mla_citation(publisher, pub_date, "location")
        elif self.citation_style == "Chicago":
            citation = self.generate_chicago_citation(publisher, pub_date)
        elif self.citation_style == "APA":
            citation = self.generate_apa_citation(pub_date, publisher)
        else:
            citation = "Invalid citation style."

        return citation, []

    def load_file(self):
        file_types = [("PDF Files", "*.pdf"), ("EPUB Files", "*.epub")]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if file_path:
            if file_path.endswith('.pdf'):
                self.extract_text_from_pdf(file_path)
            elif file_path.endswith('.epub'):
                self.extract_text_from_epub(file_path)
            else:
                with open(file_path, 'r') as file:
                    file_contents = file.read()
                    self.set_kb(file_contents)
            messagebox.showinfo("File Loaded", "File loaded successfully.")
        else:
            messagebox.showinfo("File Not Loaded", "No file selected.")


    def load_url(self):
        url = simpledialog.askstring("Load URL", "Enter the URL:")
        if url:
            self.extract_text_from_url(url)
            messagebox.showinfo("URL Loaded", "URL loaded successfully.")
        else:
            messagebox.showinfo("URL Not Loaded", "No URL entered.")

    def delete_file(self):
        self.kb = None
        messagebox.showinfo("File Deleted", "File deleted successfully.")

class Application:
    def __init__(self, master):
        self.master = master
        self.academic_writer = AcademicWriter()
        self.create_widgets()
        self.conversation = []

    def create_widgets(self):
        # Create the input area
        self.input_area = tk.Text(self.master, height=10, width=50)
        self.input_area.pack()

        # Create the citation style dropdown
        self.citation_style_var = tk.StringVar(self.master)
        self.citation_style_var.set("apa")
        self.citation_style_dropdown = tk.OptionMenu(self.master, self.citation_style_var, "apa", "mla", "chicago")
        self.citation_style_dropdown.pack()

        # Create the temperature slider
        self.temperature_var = tk.DoubleVar(self.master)
        self.temperature_var.set(0.5)
        self.temperature_slider_label = tk.Label(self.master, text="Temperature")
        self.temperature_slider_label.pack()
        self.temperature_slider = tk.Scale(self.master, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.temperature_var)
        self.temperature_slider.pack()

        # Create the token size slider
        self.token_size_var = "**********"
        self.token_size_var.set(50)
        self.token_size_slider_label = "**********"="Token Size")
        self.token_size_slider_label.pack()
        self.token_size_slider = "**********"=10, to=100, orient=tk.HORIZONTAL, variable=self.token_size_var)
        self.token_size_slider.pack()

        # Create the NER weight slider
        self.ner_weight_var = tk.DoubleVar(self.master)
        self.ner_weight_var.set(0.5)
        self.ner_weight_slider_label = tk.Label(self.master, text="NER Weight")
        self.ner_weight_slider_label.pack()
        self.ner_weight_slider = tk.Scale(self.master, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.ner_weight_var)
        self.ner_weight_slider.pack()

        # Create the "Generate" button
        self.generate_button = tk.Button(self.master, text="Generate", command=self.generate_text)
        self.generate_button.pack()

        # Create the output area
        self.output_area = tk.Text(self.master, height=10, width=50)
        self.output_area.pack()

        # Create the file upload button
        self.file_upload_button = tk.Button(self.master, text="Upload File", command=self.load_file)
        self.file_upload_button.pack()

        # Create the URL button
        self.url_button = tk.Button(self.master, text="Load URL", command=self.load_url)
        self.url_button.pack()

        # Create the status label
        self.status_label = tk.Label(self.master, text="")
        self.status_label.pack()

        # Create the citation label
        self.citation_label = tk.Label(self.master, text="")
        self.citation_label.pack()

    def load_file(self):
        file_types = [("PDF Files", "*.pdf"), ("EPUB Files", "*.epub")]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if file_path:
            self.academic_writer.kb = file_path
            messagebox.showinfo("File Loaded", "File loaded successfully.")
        else:
            messagebox.showinfo("File Not Loaded", "No file selected.")

    def load_url(self):
        url = simpledialog.askstring("Load URL", "Enter the URL:")
        if url:
            self.academic_writer.kb = url
            messagebox.showinfo("URL Loaded", "URL loaded successfully.")
        else:
            messagebox.showinfo("URL Not Loaded", "No URL entered.")

    def delete_file(self):
        self.academic_writer.kb = None
        messagebox.showinfo("File Deleted", "File deleted successfully.")

    def get_chat_history(self):
        chat_history = [message["content"] for message in self.conversation]
        return "\n".join(chat_history)

    def generate_text(self):
        # Retrieve the user input from the input area
        prompt = self.input_area.get("1.0", "end-1c")
        citation_style = self.citation_style_var.get()
        temperature = self.temperature_var.get()
        max_tokens = "**********"
        ner_weight = self.ner_weight_var.get()

        # Add the user input to the conversation
        user_input = {"role": "user", "content": prompt}
        self.conversation.append(user_input)

        # Retrieve the chat history
        chat_history = self.get_chat_history()

        # Split the chat history into chunks of 4097 tokens or less
        chunks = [chat_history[i:i+4097] for i in range(0, len(chat_history), 4097)]

        # Initialize the response variable to an empty string
        response = ""

        # Generate the response for each chunk and concatenate the responses
        for chunk in chunks:
            print("Current chunk:", chunk)  # Print the current chunk
            chunk_response = "**********"=temperature, max_tokens=max_tokens, ner_weight=int(self.ner_weight_var.get()))
            if chunk_response is not None and chunk_response != 'No knowledge base loaded.':
                response += chunk_response
                print("Current response:", response)  # Print the current response
            else:
                print("Chunk response is None or 'No knowledge base loaded.'")  # Print a message for debugging purposes

        # Append the response to the conversation
        assistant_response = {"role": "assistant", "content": response}
        self.conversation.append(assistant_response)

        # Display the conversation in the output area
        self.display_conversation()

    def display_conversation(self):
        # Clear the output area
        self.output_area.delete("1.0", "end")

        # Display the conversation in the output area
        for message in self.conversation:
            role = message["role"]
            content = message["content"]
            self.output_area.insert("end", f"{role}: {content}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()


        



