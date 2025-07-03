#date: 2025-07-03T16:48:45Z
#url: https://api.github.com/gists/8d6036869641c5c70a874cfcd11460f7
#owner: https://api.github.com/users/jtandavala

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json
from datetime import datetime
import logging
import torch
from threading import Thread
import time

# === CONFIGURATION ===
app = Flask(__name__)
app.config['SECRET_KEY'] = "**********"
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # Using smaller 7B version for local
CACHE_DIR = "./model_cache"  # Directory to cache downloaded models

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

# === RAG SYSTEM CLASS ===
class LocalLlamaRAG:
    def __init__(self):
        """Initialize RAG system with Llama model for local deployment"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Configure quantization for memory efficiency
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=self.device,
            cache_folder=CACHE_DIR
        )
        
        # Load Llama tokenizer and model
        logger.info("Loading Llama model...")
        self.tokenizer = "**********"
            LLM_MODEL,
            cache_dir=CACHE_DIR,
            use_auth_token= "**********"
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR,
            use_auth_token= "**********"
        )
        
        # Initialize document storage
        self.documents = []
        self.document_metadata = []
        self.embeddings = None
        self.index = None
        
        logger.info("RAG system initialized successfully")
    
    def add_documents(self, documents, metadata=None):
        """Add documents to the knowledge base"""
        if not documents:
            return False
        
        logger.info(f"Adding {len(documents)} documents to knowledge base")
        
        # Add documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.document_metadata.extend(metadata)
        else:
            self.document_metadata.extend([{"source": "manual", "timestamp": datetime.now().isoformat()}] * len(documents))
        
        # Generate embeddings
        new_embeddings = self.embedding_model.encode(documents, convert_to_tensor=True)
        new_embeddings = new_embeddings.cpu().numpy()
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Update FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"Total documents in knowledge base: {len(self.documents)}")
        return True
    
    def retrieve_documents(self, query, k=3):
        """Retrieve relevant documents"""
        if self.index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy()
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.document_metadata[idx],
                    'score': float(distance),
                    'rank': i + 1
                })
        
        return results
    
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"_ "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"q "**********"u "**********"e "**********"r "**********"y "**********", "**********"  "**********"c "**********"o "**********"n "**********"t "**********"e "**********"x "**********"t "**********"_ "**********"d "**********"o "**********"c "**********"s "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"n "**********"e "**********"w "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"= "**********"1 "**********"5 "**********"0 "**********") "**********": "**********"
        """Generate response using Llama with retrieved context"""
        if not context_docs:
            return "I don't have enough information to answer your question."
        
        # Build context
        context = "\n".join([doc['document'] for doc in context_docs[:2]])
        
        # Create prompt in Llama 2 chat format
        prompt = f"""<s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Answer the question using the provided context.
        <</SYS>>

        Context: {context}

        Question: {query} [/INST]"""
        
        # Tokenize and generate
        inputs = "**********"="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens= "**********"
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id= "**********"
            )
        
        response = "**********"=True)
        
        # Extract just the assistant's response
        response = response.split("[/INST]")[-1].strip()
        
        # Clean up response
        response = response.split("<|endoftext|>")[0]
        response = response.split("</s>")[0]
        
        return response or "I couldn't generate a clear answer based on the available information."
    
    def ask(self, query, k=3):
        """Complete RAG query processing"""
        try:
            # Retrieve relevant documents
            relevant_docs = self.retrieve_documents(query, k)
            
            if not relevant_docs:
                return {
                    'query': query,
                    'answer': "I don't have any relevant information to answer your question.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Generate response
            answer = self.generate_response(query, relevant_docs)
            
            # Calculate confidence based on retrieval scores
            avg_score = sum(doc['score'] for doc in relevant_docs) / len(relevant_docs)
            confidence = max(0.0, min(1.0, 1.0 - (avg_score / 10.0)))  # Normalize to 0-1
            
            return {
                'query': query,
                'answer': answer,
                'sources': [doc['document'][:200] + "..." if len(doc['document']) > 200 else doc['document'] for doc in relevant_docs],
                'confidence': confidence,
                'retrieved_docs': relevant_docs
            }
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'query': query,
                'answer': "Sorry, I encountered an error while processing your question.",
                'sources': [],
                'confidence': 0.0
            }

# === FLASK ROUTES ===
rag_system = None
model_loaded = False

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'ready' if model_loaded else 'loading',
        'model_loaded': model_loaded,
        'documents_count': len(rag_system.documents) if rag_system else 0,
        'device': str(rag_system.device) if rag_system else 'unknown'
    })

@app.route('/api/ask', methods=['POST'])
def api_ask():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded yet'}), 503
    
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400
    
    question = data['question'].strip()
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400
    
    try:
        result = rag_system.ask(question)
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in API ask: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/add_document', methods=['POST'])
def api_add_document():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded yet'}), 503
    
    data = request.get_json()
    if not data or 'document' not in data:
        return jsonify({'error': 'Document text is required'}), 400
    
    document = data['document'].strip()
    if not document:
        return jsonify({'error': 'Document cannot be empty'}), 400
    
    try:
        success = rag_system.add_documents([document])
        if success:
            return jsonify({
                'success': True,
                'message': 'Document added successfully',
                'total_documents': len(rag_system.documents)
            })
        else:
            return jsonify({'error': 'Failed to add document'}), 500
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/documents')
def api_documents():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded yet'}), 503
    
    try:
        documents = []
        for i, (doc, meta) in enumerate(zip(rag_system.documents, rag_system.document_metadata)):
            documents.append({
                'id': i,
                'text': doc[:200] + "..." if len(doc) > 200 else doc,
                'full_text': doc,
                'metadata': meta
            })
        
        return jsonify({
            'success': True,
            'documents': documents,
            'total': len(documents)
        })
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/manage')
def manage():
    return render_template('manage.html', model_loaded=model_loaded)

# === INITIALIZATION ===
def initialize_rag():
    global rag_system, model_loaded
    
    try:
        logger.info("Starting RAG system initialization...")
        rag_system = LocalLlamaRAG()
        
        # Add sample documents
        sample_docs = [
            "Flask is a lightweight web framework for Python that makes it easy to build web applications.",
            "Llama 2 is a collection of pretrained and fine-tuned generative text models from Meta.",
            "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation.",
            "Hugging Face provides tools to easily download and use pretrained models.",
            "Quantization reduces model size and memory requirements by using lower precision numbers."
        ]
        rag_system.add_documents(sample_docs)
        
        model_loaded = True
        logger.info("RAG system initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")

# === MAIN EXECUTION ===
if __name__ == '__main__':
    print("🚀 Starting Flask + Llama RAG Application (Local)")
    print("="*50)
    print("⚠️  Note: "**********"
    print("       Set it as environment variable: "**********"
    print("="*50)
    
    # Start initialization in background
    init_thread = Thread(target=initialize_rag)
    init_thread.daemon = True
    init_thread.start()
    
    print("📝 Application starting...")
    print("🌐 Web interface will be available at: http://localhost:5000")
    print("📡 API endpoints available at: /api/ask, /api/add_document, /api/documents")
    print("="*50)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)