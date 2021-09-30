#date: 2021-09-30T16:59:11Z
#url: https://api.github.com/gists/d579f5fa9a098c265cd02b8be1f924ef
#owner: https://api.github.com/users/sra84

import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

df = pd.read_csv(r'E:\Rizwan\Machine learning\29 sept assmnt\CarPrice_Assignment.csv')

X = df[['wheelbase','carlength','curbweight','enginesize','horsepower','peakrpm']]
y=df['price']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX,y)

scaled = scale.transform([[111.27,190,3000,147, 119,5700]])
predictedPrice=regr.predict(scaled)
print(predictedPrice)