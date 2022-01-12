#date: 2022-01-12T17:17:15Z
#url: https://api.github.com/gists/f3ea5a082e05fbeeaee68a9b0dfe75e3
#owner: https://api.github.com/users/haykaza

#drop correlated variables
X = all_data.drop(columns = ['Instrument', 'Label', 'AD', 'Operating Margin - %, TTM','EV to EBITDA', 
                             'Price To Sales Per Share (Daily Time Series Ratio)'])
y = all_data['Label']

#create separate datasets for no sentiment and sentiment based data
X_NoSent = X.drop(columns = ['SentOverallLabs', 'SentOverallFBert'])
X_BertRna = X.drop(columns = ['SentOverallFBert'])
X_Finbert = X.drop(columns = ['SentOverallLabs'])