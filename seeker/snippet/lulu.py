#date: 2024-08-20T17:06:22Z
#url: https://api.github.com/gists/a1eabdbf295b1a69733f86d5f3c52e20
#owner: https://api.github.com/users/eduluzufop

import json
import torch
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments

# Verificar se GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paragrafos_espiritas = []

# Carregar o arquivo 'paragrafos_livrosKardec.json'
with open('paragrafos_livrosKardec.json', 'r', encoding='utf-8') as f:
    paragrafos = json.load(f)

# Atualizar cada parágrafo com a propriedade 'textoOriginal'
for par_id, par_data in tqdm(paragrafos.items(), desc="Atualizando parágrafos", unit="parágrafo"):
    text = par_data['textoOriginal']
    categoriaDoTexto = par_data['categoriaDoTexto']
    if categoriaDoTexto in ['nota', 'item', 'texto']:
        paragrafos_espiritas.append(text)

# Criar um DataFrame com os parágrafos
df = pd.DataFrame(paragrafos_espiritas, columns=['text'])
df.to_csv('dataFrameKardec.csv', index=False)
dataset = Dataset.from_pandas(df)

# Carregar o tokenizer e modelo BERT
tokenizer = "**********"
model = AutoModelForMaskedLM.from_pretrained('neuralmind/bert-large-portuguese-cased').to(device)

# Função de tokenização e aplicação de máscara
 "**********"d "**********"e "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"_ "**********"f "**********"u "**********"n "**********"c "**********"t "**********"i "**********"o "**********"n "**********"( "**********"e "**********"x "**********"a "**********"m "**********"p "**********"l "**********"e "**********"s "**********") "**********": "**********"
    outputs = "**********"="max_length", truncation=True, max_length=512, return_tensors="pt")

    # Movendo os tensores para a GPU
    outputs = {key: value.to(device) for key, value in outputs.items()}
    
    # Garantir que 'labels' seja uma cópia de 'input_ids'
    outputs["labels"] = outputs["input_ids"].clone()

    # Criar máscaras aleatórias para a tarefa de MLM
    probability_matrix = torch.full(outputs["labels"].shape, 0.15, device=device)
    special_tokens_mask = "**********"
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens= "**********"
    ]
    special_tokens_mask = "**********"=torch.bool, device=device)
    probability_matrix.masked_fill_(special_tokens_mask, value= "**********"
    masked_indices = torch.bernoulli(probability_matrix).bool()
    outputs["labels"][~masked_indices] = "**********"

    # Máscara os tokens que foram marcados
    outputs["input_ids"][masked_indices] = "**********"

    return {key: value.cpu().numpy() for key, value in outputs.items()}  # Retorna os dados para CPU para evitar problemas de incompatibilidade

# Aplicar a tokenização com as máscaras
tokenized_dataset = "**********"=True)

# Configurar os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./bert-finetuned-espirita",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    resume_from_checkpoint=True,
)

# Configurar o trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= "**********"
    eval_dataset= "**********"
)

# Iniciar o treinamento
trainer.train()
