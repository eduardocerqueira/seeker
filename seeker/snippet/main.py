#date: 2023-11-22T17:01:46Z
#url: https://api.github.com/gists/5fefb565b286b984097cb59184c0fdcc
#owner: https://api.github.com/users/pulkit-30

from transformers import pipeline
import PyPDF2
from docx import Document
import os



# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from Word document
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text
    return text

# Load the question-answering pipeline from Hugging Face's Transformers
qa_pipeline = pipeline("question-answering")

# Read the document
file_path = '/Users/pulkitgupta/Desktop/temp/Final Proposal for GSOC.pdf'  # Replace this with your document path
if file_path.endswith('.pdf'):
    text = extract_text_from_pdf(file_path)
elif file_path.endswith('.docx'):
    text = extract_text_from_docx(file_path)
else:
    print("Unsupported file format")

# Define a question
question = "Who is the author of this proposal?"

# Get the answer using the AI model
answer = qa_pipeline(question=question, context=text)

print(f"Question: {question}")
print(f"Answer: {answer['answer']}")
print(f"Confidence: {answer['score']}")
