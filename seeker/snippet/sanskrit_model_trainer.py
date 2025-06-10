#date: 2025-06-10T16:54:35Z
#url: https://api.github.com/gists/7b76d2efba8eeafc24f0b907d01687cb
#owner: https://api.github.com/users/neilh44

#!/usr/bin/env python3
"""
Sanskrit Language Model Trainer - Atharvashira Upanishad
Train transformer model using existing tokenizer
"""

import os
import pickle
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from collections import Counter
import math

class SanskritDataset(Dataset):
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"_ "**********"i "**********"n "**********"i "**********"t "**********"_ "**********"_ "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"s "**********"e "**********"q "**********"u "**********"e "**********"n "**********"c "**********"e "**********"s "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********") "**********": "**********"
        self.sequences = sequences
        self.tokenizer = "**********"
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SanskritTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool()
        mask = mask.to(x.device)
        
        x = self.transformer(x, mask=mask)
        return self.output_proj(x)

class SanskritTrainer:
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"_ "**********"i "**********"n "**********"i "**********"t "**********"_ "**********"_ "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"_ "**********"p "**********"a "**********"t "**********"h "**********"= "**********"" "**********"f "**********"a "**********"s "**********"t "**********"_ "**********"d "**********"e "**********"v "**********"a "**********"n "**********"a "**********"g "**********"a "**********"r "**********"i "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********". "**********"p "**********"k "**********"l "**********"" "**********") "**********": "**********"
        print("🔮 SANSKRIT MODEL TRAINER - ATHARVASHIRA UPANISHAD")
        print("="*60)
        
        # Load tokenizer
        print(f"📚 Loading tokenizer: "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"_ "**********"p "**********"a "**********"t "**********"h "**********", "**********"  "**********"' "**********"r "**********"b "**********"' "**********") "**********"  "**********"a "**********"s "**********"  "**********"f "**********": "**********"
            tokenizer_data = "**********"
        
        self.token_to_id = "**********"
        self.id_to_token = "**********"
        self.vocab_size = "**********"
        
        print(f"✅ Tokenizer loaded: "**********":,} vocab")
        
        # Training config
        self.seq_len = 128
        self.batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Device: {self.device}")
        
    def is_devanagari(self, char):
        return 0x0900 <= ord(char) <= 0x097F
    
    def clean_text(self, text):
        cleaned = ''.join(c for c in text if self.is_devanagari(c) or c in ' \t\n।॥')
        return re.sub(r'\s+', ' ', cleaned).strip()
    
    def encode_text(self, text):
        text = self.clean_text(text)
        words = re.findall(r'[^\s।॥,;:!?\.\(\)\[\]\"\'`]+', text)
        
        token_ids = "**********"
        for word in words:
            tokens = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
                token_ids.append(self.token_to_id.get(token, self.token_to_id['<UNK>']))
        return token_ids
    
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
        tokens = "**********"
        i = 0
        while i < len(word):
            found = False
            for j in range(len(word), i, -1):
                subword = word[i:j]
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"u "**********"b "**********"w "**********"o "**********"r "**********"d "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"t "**********"o "**********"_ "**********"i "**********"d "**********": "**********"
                    tokens.append(subword)
                    i = j
                    found = True
                    break
            if not found:
                tokens.append(word[i] if word[i] in self.token_to_id else '<UNK>')
                i += 1
        return tokens
    
    def load_corpus(self, corpus_file="ath_upn.txt"):
        print(f"📖 Loading corpus: {corpus_file}")
        
        if not os.path.exists(corpus_file):
            print(f"❌ File not found: {corpus_file}")
            return None
            
        with open(corpus_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean and tokenize
        cleaned_text = self.clean_text(content)
        print(f"📊 Original: {len(content):,} chars, Cleaned: {len(cleaned_text):,} chars")
        
        # Encode to token IDs
        token_ids = "**********"
        print(f"🔤 Tokenized: "**********":,} tokens")
        
        # Calculate vocab coverage
        unique_tokens = "**********"
        unk_count = "**********"
        unk_rate = "**********"
        
        print(f"📊 Unique tokens used: "**********":,}/{self.vocab_size:,}")
        print(f"❓ UNK rate: {unk_rate:.2%}")
        
        return token_ids
    
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"c "**********"r "**********"e "**********"a "**********"t "**********"e "**********"_ "**********"s "**********"e "**********"q "**********"u "**********"e "**********"n "**********"c "**********"e "**********"s "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"s "**********") "**********": "**********"
        print(f"🔢 Creating sequences (length: {self.seq_len})")
        
        sequences = []
        for i in range(0, len(token_ids) - self.seq_len, self.seq_len // 2): "**********"
            seq = token_ids[i: "**********"
            if len(seq) == self.seq_len + 1:
                sequences.append(seq)
        
        print(f"📚 Created {len(sequences):,} training sequences")
        return sequences
    
    def train_model(self, sequences, epochs=10):
        print(f"🚀 Training model...")
        print(f"📊 Sequences: {len(sequences):,} | Batch size: {self.batch_size} | Epochs: {epochs}")
        
        # Create dataset and dataloader
        dataset = SanskritDataset(sequences, self)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        model = SanskritTransformer(
            vocab_size=self.vocab_size,
            d_model=256,
            nhead=8,
            num_layers=6,
            seq_len=self.seq_len
        ).to(self.device)
        
        # Training setup
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = "**********"=self.token_to_id.get('<PAD>', 0))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        print(f"🧠 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Reshape for loss calculation
                loss = criterion(outputs.reshape(-1, self.vocab_size), targets.reshape(-1))
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Progress update
                if batch_idx % 50 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"   Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {avg_loss:.4f}")
            
            scheduler.step()
            avg_loss = total_loss / batch_count
            print(f"🔄 Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint_path = f"sanskrit_model_epoch_{epoch+1}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss,
                    'vocab_size': self.vocab_size,
                    'seq_len': self.seq_len
                }, checkpoint_path)
                print(f"💾 Saved: {checkpoint_path}")
        
        # Save final model
        final_path = "sanskrit_model_final.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': self.vocab_size,
            'seq_len': self.seq_len,
            'token_to_id': "**********"
            'id_to_token': "**********"
        }, final_path)
        print(f"✅ Final model saved: {final_path}")
        
        return model
    
    def generate_text(self, model, prompt="ॐ", max_length=100, temperature=0.8):
        print(f"\n🎭 Generating text...")
        print(f"🌱 Prompt: '{prompt}'")
        
        model.eval()
        
        # Encode prompt
        token_ids = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"s "**********": "**********"
            token_ids = "**********"
        
        input_ids = "**********"=torch.long).to(self.device)
        
        generated = "**********"
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input (last seq_len tokens)
                if input_ids.size(1) > self.seq_len:
                    current_input = input_ids[:, -self.seq_len:]
                else:
                    current_input = input_ids
                
                # Get predictions
                outputs = model(current_input)
                next_token_logits = outputs[0, -1, : "**********"
                
                # Apply softmax and sample
                probs = "**********"=-1)
                next_token = "**********"
                
                # Stop at sentence end
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"e "**********"x "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"= "**********"= "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"t "**********"o "**********"_ "**********"i "**********"d "**********". "**********"g "**********"e "**********"t "**********"( "**********"' "**********"। "**********"< "**********"/ "**********"w "**********"> "**********"' "**********", "**********"  "**********"- "**********"1 "**********") "**********": "**********"
                    generated.append(next_token)
                    break
                
                generated.append(next_token)
                input_ids = "**********"=1)
        
        # Decode
        generated_tokens = "**********"
        generated_text = "**********"
        generated_text = re.sub(r'\s+', ' ', generated_text).strip()
        
        print(f"📜 Generated: '{generated_text}'")
        return generated_text
    
    def test_model(self, model):
        print(f"\n🧪 Testing model with sample prompts...")
        
        test_prompts = [
            "ॐ",
            "ब्रह्म",
            "आत्मा",
            "सत्यम्",
            "शिवम्"
        ]
        
        for prompt in test_prompts:
            self.generate_text(model, prompt, max_length=50, temperature=0.7)
            print()

def main():
    # Find tokenizer
    tokenizer_files = "**********"
    tokenizer_path = "**********"
    
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"f "**********"i "**********"l "**********"e "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"_ "**********"f "**********"i "**********"l "**********"e "**********"s "**********": "**********"
        if os.path.exists(file):
            tokenizer_path = "**********"
            break
    
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"_ "**********"p "**********"a "**********"t "**********"h "**********": "**********"
        print("❌ No tokenizer found! Train tokenizer first.")
        return
    
    # Initialize trainer
    trainer = "**********"
    
    # Load corpus
    token_ids = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"s "**********": "**********"
        return
    
    # Create sequences
    sequences = "**********"
    if len(sequences) < 10:
        print("❌ Not enough sequences for training!")
        return
    
    # Train model
    model = trainer.train_model(sequences, epochs=20)
    
    # Test generation
    trainer.test_model(model)
    
    print("\n✅ Training complete!")
    print("🎯 Use 'sanskrit_model_final.pt' for text generation")

if __name__ == "__main__":
    main()