#date: 2022-07-19T17:03:02Z
#url: https://api.github.com/gists/fb4ba2565686238c0d9236cc5782ece7
#owner: https://api.github.com/users/AparnaDhinakaran

def get_embeddings(batch: Dict) -> Dict:
   inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}
   with torch.no_grad():
       out = model(**inputs)
       # (layer_#, batch_size, seq_length/or/num_tokens, hidden_size)
       hidden_states = torch.stack(out.hidden_states)  
       embeddings = hidden_states[-1][:,0,:] # Select last layer, then CLS token vector
      
   return {"text_vector": embeddings.cpu().numpy()}