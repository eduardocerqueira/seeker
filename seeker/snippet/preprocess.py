#date: 2021-08-31T13:14:27Z
#url: https://api.github.com/gists/4c438df950d5a314036dc3d96ec9bc88
#owner: https://api.github.com/users/Ashcom-git

def preprocess(text):

  text_token = CountVectorizer().build_tokenizer()(text.lower())
  text_token = [lemma.lemmatize(i) for i in text_token if i not in stop]
  if len(text_token)>0:
    return ' '.join(text_token)+'\n'
  return '\n'  

def complete_preprocess(list_):

  processed_lst = []
  for i in list_:
    pre = preprocess(i)
    processed_lst.append(pre) 
 
  return processed_lst