#date: 2022-02-15T17:02:11Z
#url: https://api.github.com/gists/573c2f861aa7a256d712466785be3baa
#owner: https://api.github.com/users/vaibhavtmnit

from transformers import BertTokenizer
from functools import partial
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def collate_fn_bert(tokenizer,batch):
    
    # Dividing list of X,y into lists of X & y   
    X, y = list(zip(*batch))
    
    # Try checking content of X & y before next step
    
    X = tokenizer(text=X, padding=True, return_tensors='pt')
    
    return X,torch.tensor(y)
    
    
class TextDataset(Dataset):

    def __init__(self,x,y):
        
        self.x = x
        self.y = y
  
           
    def __len__(self):
    
        return len(self.x)   
    
    def __getitem__(self, i):
        
        # Instead of returning tensor we will now return raw text as tokenization will be performed by collate_fn
        return self.x[i],self.y[i]  
                 
textdataset = TextDataset(text_data,sentiment)
