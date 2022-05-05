#date: 2022-05-05T17:17:27Z
#url: https://api.github.com/gists/13d2e47c76c7cfbf7a975b8ab924d4ba
#owner: https://api.github.com/users/huzefasr

#BDA CODES

#WORD COUNT
file = open(r"D:\BDA\sample.txt", "r")
read_data = file.read()
per_word = read_data.split()

print('Total Words:', len(per_word))

#MATRIX_MUL
import numpy as np
a=np.array([[1,0],[0,1]])
b=np.array([[4,1],[2,2]])
np.matmul(a,b)

#Apriori
import numpy as np
import pandas as pd
from apyori import apriori
store_data=pd.read_csv('D:\BDA\dataset_apriori.csv',header=None)
store_data
records=[]
for i in range(0,22):
    records.append([str(store_data.values[i,j]) for j in range(0,6)])

association_rules=apriori(records,min_support=0.5,min_confidence=0.7,min_lift=1.2,min_length=2)
association_results=list(association_rules)

"""
#Classification
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('D:\BDA\iris.csv')
X = data.iloc[:,:-1].values
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train, y_train)
KNN_prediction = KNN_model.predict(X_test)
print(accuracy_score(KNN_prediction, y_test))
print(classification_report(KNN_prediction, y_test))

"""

#PAGE RANK
import networkx as nx
G=nx.barbell_graph(5,1)
pr=nx.pagerank(G,0.4)
pr
nx.draw(G,with_labels=True)
"""