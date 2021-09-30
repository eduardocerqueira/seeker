#date: 2021-09-30T17:01:50Z
#url: https://api.github.com/gists/739037f9946ea12c6172b493175cd57d
#owner: https://api.github.com/users/sra84

import pandas as pd
from sklearn import linear_model

df = pd.read_csv(r'E:\Rizwan\Machine learning\29 sept assmnt\diabetes.csv')
X= df[['Glucose', 'Insulin', 'DiabetesPedigreeFunction']]
y=df['Outcome']
regr = linear_model.LinearRegression()
regr.fit(X,y)

diabetic = regr.predict([[100, 0,1.2]])

if diabetic> 0.5:
    print("person is diabetic")
else:
    print("not diabetic")
