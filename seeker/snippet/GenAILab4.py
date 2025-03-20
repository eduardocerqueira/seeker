#date: 2025-03-20T16:46:38Z
#url: https://api.github.com/gists/3c4e78d2949ef228c3a26ba0a99e74b4
#owner: https://api.github.com/users/plushycat

# Install required libraries 
# Install gensim for downloading pre-trained models
%pip install gensim
# Install Hugging Face Transformers for NLP pipelines
%pip install transformers  
# Install NLTK for text preprocessing and tokenization
%pip install nltk  

# % is used instead of ! because in some code editors/notebooks the ! is not supported.

# Import libraries
import gensim.downloader as api  
from transformers import pipeline  
import nltk  
import string  
from nltk.tokenize import word_tokenize  

# Download the 'punkt' resource from NLTK
nltk.download('punkt')
nltk.download('punkt_tab')  # Explicitly download the punkt_tab resource

# Load pre-trained word vectors
print("Loading pre-trained word vectors...")
word_vectors = api.load("glove-wiki-gigaword-100")  # Load Word2Vec model  

# Function to replace words in the prompt with their most similar words
def replace_keyword_in_prompt(prompt, keyword, word_vectors, topn=1):
    """
    Replace only the specified keyword in the prompt with its most similar word.

    Args:
        prompt (str): The original input prompt.
        keyword (str): The word to be replaced with a similar word.
        word_vectors (gensim.models.KeyedVectors): Pre-trained word embeddings.
        topn (int): Number of top similar words to consider (default: 1).

    Returns:
        str: The enriched prompt with the keyword replaced.
    """
    words = "**********"
    enriched_words = []
    
    for word in words:
        cleaned_word = word.lower().strip(string.punctuation)  # Normalize word
        try:
            if cleaned_word == keyword.lower():  # Replace only if it matches the keyword
                # Retrieve similar word
                similar_words = word_vectors.most_similar(cleaned_word, topn=topn)
                if similar_words:
                    replacement_word = similar_words[0][0]  # Choose the most similar word
                    print(f"Replacing '{word}' → '{replacement_word}'")
                    enriched_words.append(replacement_word)
                    continue  # Skip appending the original word
        except KeyError:
            print(f"'{keyword}' not found in the vocabulary. Using original word.")
        
        enriched_words.append(word)  # Keep original if no replacement was made

    enriched_prompt = " ".join(enriched_words)
    print(f"\n🔹 Enriched Prompt: {enriched_prompt}")
    return enriched_prompt

# Load an open-source Generative AI model (GPT-2)
print("\nLoading GPT-2 model...")
generator = pipeline("text-generation", model="gpt2")  

# Function to generate responses using the Generative AI model
def generate_response(prompt, max_length=100):
    try:
        response = generator(prompt, max_length=max_length, num_return_sequences=1)
        return response[0]['generated_text']
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Example original prompt
original_prompt = "Who is king."
print(f"\n🔹 Original Prompt: {original_prompt}")

# Retrieve similar words for key terms in the prompt
key_term = "king"

# Enrich the original prompt
enriched_prompt = replace_keyword_in_prompt(original_prompt, key_term, word_vectors)

# Generate responses for the original and enriched prompts
print("\nGenerating response for the original prompt...")  
original_response = generate_response(original_prompt)  
print("\nOriginal Prompt Response:")  
print(original_response)  

print("\nGenerating response for the enriched prompt...")  
enriched_response = generate_response(enriched_prompt)  
print("\nEnriched Prompt Response:")  
print(enriched_response)  

# Compare the outputs
print("\nComparison of Responses:")  
print("\nOriginal Prompt Response Length:", len(original_response))  
print("Enriched Prompt Response Length:", len(enriched_response))  

print("\nOriginal Prompt Response Detail:", original_response.count("."))  
print("Enriched Prompt Response Detail:", enriched_response.count("."))  
 Detail:", enriched_response.count("."))  
