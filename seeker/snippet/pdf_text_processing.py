#date: 2024-02-13T16:59:00Z
#url: https://api.github.com/gists/1a1ea581f71a7d0dca23dd2529b20308
#owner: https://api.github.com/users/jackhmiller

import PyPDF2
import os
import re
from typing import Callable, List, Tuple, Dict
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


def parse_pdf(file_path: str) -> List[Tuple[int, str]]:
  """
  Extracts the text from each page of the PDF.

  :param file_path: The path to the PDF file.
  :return: A list of tuples containing the page number and the extracted text.
  """
  with open('sample.pdf', 'rb') as file:
      pdf_reader = PyPDF2.PdfFileReader(file)
      num_pages = pdf_reader.numPages

      pages = []

      for i in range(num_pages):

          page = pdf_reader.getPage(i)
          text = page.extractText()
          if text.strip():  # Check if extracted text is not empty
            pages.append((page_num + 1, text))
      
      return pages

  
  def text_to_docs(text: List[str]) -> List[Document]:
    """Converts list of strings to a list of Documents with metadata."""
    doc_chunks = []

    for page_num, page in text:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-{i}",
#                    **metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks
  
 def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(
    pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]
) -> List[Tuple[int, str]]:
    cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_num, text))
    return cleaned_pages
  
  
if __name__ == '__main__':
    raw_pages = parse_pdf(file_path)

    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
    document_chunks = text_to_docs(cleaned_text_pdf)