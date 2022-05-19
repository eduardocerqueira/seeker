#date: 2022-05-19T17:21:17Z
#url: https://api.github.com/gists/23d49092993aa906ccb588d71834ff8f
#owner: https://api.github.com/users/natnondesu

import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import torch
from nltk.translate.bleu_score import corpus_bleu
from source.models.Attention import Attention
from source.utils import get_caption_back
from source.Bleuloss.expectedMultiBleu import bleu

# Test
def evaluate(encoder, decoder, device, test_loader, vocab_dict, caption):
    encoder.eval()
    decoder.eval()
    total_preds = []
    total_labels = []
    total_bleu = []

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data_img = data[0].to(device)
            data_cap = data[1].to(device)
            caption_length = data[2]
            caption_idx = data[3]
            img_latent = encoder(data_img)
            outputs, alphas = decoder.inference(img_latent)
            outputs = outputs.tolist()
            hypotheses = get_caption_back(outputs, vocab_dict)
            references = [caption[i] for i in caption_idx]
            
            bleu4 = corpus_bleu(references, hypotheses, weights=[(0.5, 0.5),(0.333, 0.333, 0.334),(0.25, 0.25, 0.25, 0.25)])
            total_bleu.append(bleu4)

    return np.array(total_bleu).mean(axis=0)
  
# IN Jupyter notebook just call . .
  
best_val = 0
os.makedirs('weights', exist_ok=True)
best_enc_path = 'weights/best_enc_path.pth' 
best_dec_path = 'weights/best_dec_path.pth' 
for epoch in range(EPOCH):
    print("On Epoch: ", epoch+1, " . . .")
    loss = train(encoder, decoder, device, train_loader, optimizer, criterion, epoch, log_interval=50)
    val_bleu2, val_bleu3, val_bleu4 = evaluate(encoder, decoder, device, val_loader, vocab_dict, val_cap)
    print(f"Average train loss: {loss} ", f"# Validation BLEU [2,3,4] : {val_bleu2}, {val_bleu3}, {val_bleu4}", "\n")
    if val_bleu4 > best_val:
        best_val = val_bleu4
        print("### New best model found, Saving model . . .")
        torch.save(encoder.state_dict(), best_enc_path)
        torch.save(decoder.state_dict(), best_dec_path)
    this_scheduler.step(np.around(val_bleu4, decimals=4))
print("\n\n #### Best BLEU score : ", best_val)