#date: 2021-09-30T17:04:18Z
#url: https://api.github.com/gists/b4793c5d1677aa598d25de6159d546b9
#owner: https://api.github.com/users/sra84

import pandas as pd
from sklearn import linear_model

df = pd.read_csv(r'E:\Rizwan\Machine learning\29 sept assmnt\USA_Housing.csv')
X= df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y=df['Price']
regr = linear_model.LinearRegression()
regr.fit(X,y)

predictedPrice = regr.predict([[50593.695,4.21032,7.331330,4.8,34467.7586]])
print(predictedPrice)
