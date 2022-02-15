#date: 2022-02-15T16:59:41Z
#url: https://api.github.com/gists/0020cae11b37eb1827d7bad9d55ece58
#owner: https://api.github.com/users/vaibhavtmnit

class TextDataset(Dataset):

    def __init__(self,x,y,vocab):
        
        self.x = x
        self.y = y
        self.vocab = vocab
           
    def __len__(self):
    
        return len(self.x)   
    
    def __getitem__(self, i):
        
        # Tokenizing and getting index of each token
        xi = torch.tensor([self.vocab[token] for token in self.x[i].split(" ")])
        yi = torch.tensor(self.y[i])

        return xi,yi
    
    
# Checking TextDataset

textdataset = TextDataset(text_data,sentiment,stoi)


for i,x in zip(textdataset,text_data):
    print(x,'\n',i,'\n ----------------------')
    
    
"""
Output 

i love pytorch 
 (tensor([0, 1, 2]), tensor(1)) 
 ----------------------
pytorch is the best 
 (tensor([2, 3, 4, 5]), tensor(1)) 
 ----------------------
there are others but nothing as good as pytorch 
 (tensor([ 6,  7,  8,  9, 10, 11, 12, 11,  2]), tensor(1)) 
 ----------------------
pytorch is future 
 (tensor([ 2,  3, 13]), tensor(1)) 
 ----------------------
viva la pytorch 
 (tensor([14, 15,  2]), tensor(1)) 
 ----------------------
pytorch rocks 
 (tensor([ 2, 16]), tensor(1)) 
 ----------------------
"""