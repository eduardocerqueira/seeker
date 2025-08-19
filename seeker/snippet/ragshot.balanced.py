#date: 2025-08-19T17:09:31Z
#url: https://api.github.com/gists/2e09d2608ddb6337989121f90dd87c7a
#owner: https://api.github.com/users/JoeCooper

n_train = 1024 # 5600
n_test = 2048 # 380000
n_examples = 2 # this is PER CLASS
batch_size = 4
model_id = "meta-llama/Llama-3.2-1B-Instruct"
from os.path import exists
from sys import stderr
def load_csv(filename: str) -> list[tuple[str, str]]:
    if not exists(filename):
        print(f"{filename} not found!", file=stderr)
    with open(filename, 'r') as f:
        import csv
        r = csv.reader(f)
        return list((row[1], row[0]) for row in r)
train = load_csv('yelp_review_polarity_csv/train.csv')[:n_train]
def enumerate_batch(l: list, n: int):
    while len(l) > 0:
        b = l[:n]
        l = l[n:]
        yield b
import torch
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
device = 'cuda'
def load(s: str) -> tuple[tensor, tensor, tensor, tensor]:
    with open(s, 'rb') as f:
        return torch.load(f, map_location=device)
tokenizer = "**********"
    model_id,
    use_fast=True)
tokenizer.pad_token = "**********"
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device)
pad_id = 2
model.eval()
def embed_batch(
    input_ids = torch.empty(1, 0),
    attention_mask = torch.empty(1, 0)) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        attention_mask_expanded = attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9) # Avoid division by zero
        batch_embeddings = sum_embeddings / sum_mask
        return batch_embeddings
d_embeddings = "**********"
embeddings = torch.zeros([0, d_embeddings]).to(device)
for batch in enumerate_batch(train, batch_size):
    stimulii = [x[0] for x in batch]
    tokenized = "**********"
        stimulii,
        return_tensors="pt",
        padding=True)
    tokenized.to(device)
    batch_embeddings = "**********"
    embeddings = torch.cat([embeddings, batch_embeddings], dim=0)
    print(f"[embedding] {str(embeddings.shape)}")
import hnswlib
embeddings = F.normalize(embeddings, p=2, dim=1)
n_embeddings = embeddings.shape[0]
index_by_class = {}
emission_whitelist = list(set(x[1] for x in train))
for classification in emission_whitelist:
    indices = [x for x in range(n_embeddings) if train[x][1].strip() == classification]
    shortlist = torch.zeros([0, d_embeddings]).to(device)
    for x in indices:
        e = embeddings[x, ]
        e = e.unsqueeze(dim=0)
        shortlist = torch.cat([shortlist, e], dim=0)
    vectors = shortlist.cpu().numpy()
    index = hnswlib.Index(space='l2', dim=d_embeddings)
    index.init_index(max_elements=len(indices), ef_construction=100, M=16)
    index.add_items(vectors, indices)
    index.set_ef(50)
    index_by_class[classification] = index
    print(f"[embedding] `{classification}` complete; {shortlist.shape[0]} records.")
test = load_csv('yelp_review_polarity_csv/test.csv')[:n_test]
instruction = 'Classify the review as either `1` (bad) or `2` (good). Write nothing else.'
points = 0
eot_id = "**********"
progress = 0
token_whitelist = "**********"=False) + tokenizer.encode('2', add_special_tokens=False)
embedding_matrix = "**********"
v = embedding_matrix.shape[0]
logit_mask = torch.tensor([0.0] * v).to(device)
 "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"w "**********"h "**********"i "**********"t "**********"e "**********"l "**********"i "**********"s "**********"t "**********": "**********"
    logit_mask[i] = 1.0
for batch in enumerate_batch(test, batch_size):
    stimulii = [x[0] for x in batch]
    tokenized = "**********"
        stimulii,
        return_tensors='pt',
        padding=True)
    tokenized.to(device)
    batch_embeddings = "**********".cpu()
    prompts: list[str] = []
    for i in range(len(stimulii)):
        stimulus = stimulii[i]
        query = batch_embeddings[i,]
        ex_keys = []
        ex_values = []
        for emission in emission_whitelist:
            index = index_by_class[emission]
            labels, distances = index.knn_query(
                query,
                k=n_examples)
            l = labels.squeeze(0).tolist()
            ex_keys = ex_keys + [train[x][0] for x in l]
            ex_values = ex_values + [train[x][1] for x in l]
        ex_renders = [f"Example Review: `{ex_keys[i]}`\nSentiment: `{ex_values[i]}`" for i in range(len(ex_keys))]
        ex_render = '\n\n'.join(ex_renders)
        ex_shunt = '\n\n' if len(ex_renders) > 0 else ''
        prompt = f"Task: {instruction}{ex_shunt}{ex_render}\n\nReview: `{stimulus}`\nSentiment: `"
        prompts.append(prompt)
    prompts_tokenized = "**********"
        prompts,
        return_tensors='pt',
        add_special_tokens= "**********"
        padding=True).to(device)
    with torch.no_grad():
        outputs = "**********"
    input_ids = "**********"
    attention_mask = "**********"
    lengths = attention_mask.sum(dim=1)
    last_token_indices = "**********"
    batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
    logits = outputs.logits[batch_indices, last_token_indices, : "**********"
    b = len(batch)
    masked_logits = logits * logit_mask.expand(b, -1)
    values, indices = masked_logits.max(dim=-1)
    targets = [x[1] for x in batch]
    tokenized_targets = "**********"="pt", add_special_tokens=False)['input_ids'].squeeze(1).tolist()
    matches = "**********"== x[1] for x in zip(indices, tokenized_targets)]
    count = len([x for x in matches if x])
    print(f"{indices} vs. {tokenized_targets}; {count}; {progress} / {len(test)}")
    points = points + count
    progress += len(batch)
percent = int(100 * points / len(test))
print(f"[total] {points} / {len(test)}; {percent}%")