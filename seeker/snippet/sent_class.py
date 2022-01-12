#date: 2022-01-12T17:10:35Z
#url: https://api.github.com/gists/9c5866d26f5723854912dc2556ac9f92
#owner: https://api.github.com/users/haykaza

#create empty dictionary to store the sentiment values
sentiments = {'RIC':[],'SentOverallFBert':[],'SentOverallLabs':[]}

#loop over each RIC
for ric in dfs:
    #append the RIC to the dictionary
    sentiments['RIC'].append(ric)
    
    #initiate overall Sentiment for FinBert
    SentOverallFBert = 0
    #loop over each news headline belonging to the RIC
    for text in dfs[ric]['Headlines']:
        #tokenize the headlines
        inputs = tokenizer(text, return_tensors="pt")
        #get prediction outputs
        outputs = model(**inputs)
        #get the maximum probability class
        sentF = torch.argmax(outputs[0])
        #update SentOverallFBert based on the classification output
        if sentF == 0:
            SentOverallFBert += 1
        elif sentF == 1:
            SentOverallFBert -= 1
    #append FinBert calculated overall sentiment of a company to the dictionary            
    sentiments['SentOverallFBert'].append(SentOverallFBert)
    
    #initiate overall Sentiment for BERT-RNA
    SentOverallLabs = 0
    #update SentOverallLabs based on the classification output
    for sentL in dfs[ric]['sentimentLabs']:
        if sentL == 0:
            SentOverallLabs += 1
        elif sentL == 1:
            SentOverallLabs -= 1 
    #append BERT-RNA calculated overall sentiment of a company to the dictionary            
    sentiments['SentOverallLabs'].append(SentOverallLabs)
    
#convert dictionary to a dataframe
sentiments = pd.DataFrame(sentiments)