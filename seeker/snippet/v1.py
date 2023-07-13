#date: 2023-07-13T16:58:27Z
#url: https://api.github.com/gists/814301bc7e7d6ab1616e9a0a9875e268
#owner: https://api.github.com/users/akashjassal3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#loading dataset
df=pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
df.head()

#tabular view
tab = pd.DataFrame(df)
#print(tab)


print(df.columns)



#adding BMI
df['BMI'] = df['Weight']/(df['Height']**2)

#reordering columns to put BMI
df = df[['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad', 'BMI']]

#checking whether bmi is added
df.head()

#shaoe of dataset/ printing data coloums after adding
df.shape

print('\nAfter adding BMI column\n\n',df.columns)

#shape of data
print(df.shape)

#checking duplicate rows
dup_data = df[df.duplicated()]
print(dup_data.shape)

#duplicates
duplicates = df[df.duplicated()]
print("Duplicate Rows:")
print(duplicates)

#removing duplicates
df.drop_duplicates(keep='last', inplace=True)

#shape of datset
print("\nShape of data after removing duplicates:\n\n",df.shape)


df.info()

#plotting graph

fig = plt.figure(figsize = (16, 7))
counts = df["NObeyesdad"].value_counts()
plt.bar(counts.index, counts.values, color="purple")
plt.xlabel("Weight Classification")
plt.ylabel("Number of Respondents")
plt.title("Respondents V/s Weight Classification")
plt.show()