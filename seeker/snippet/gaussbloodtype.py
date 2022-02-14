#date: 2022-02-14T17:03:51Z
#url: https://api.github.com/gists/ace182997dc05768139bf1d77244075b
#owner: https://api.github.com/users/nemanjav11

import pandas as pd

df= pd.read_csv('tabela.csv')


#Calculates the mean value of not working days from "Št.. ..09" row

a= df["Število dni bolniške v 2009"].mean()


#Counts blood types and saves in 4 different groups

countA=0
countB=0
countAB=0
count0=0

for i in df["Krvna skupina"]:
    if i=="A":
        countA=countA+1
    elif i=="B":
        countB=countB+1
    elif i=="AB":
        countAB=countAB+1
    else:
        count0=count0+1

#Makes lists with blood type and with not working days
bloodType=[]
notWorkDays=[]


for i in df["Krvna skupina"]:
    bloodType.append(i)
    
for i in df["Število dni bolniške v 2009"]:
    notWorkDays.append(i)

#Separates days by 4 blood type groups

daysA=[]

daysB=[]

daysAB=[]

days0=[]
count=0
for i in bloodType:
    if i =="A":
        current=notWorkDays[count]
        count=count+1
        daysA.append(current)
        
    elif i =="B":
        current=notWorkDays[count]
        count=count+1
        daysB.append(current)
    elif i =="AB":
        current=notWorkDays[count]
        count=count+1
        daysAB.append(current)    
    elif i =="0":
        current=notWorkDays[count]
        count=count+1
        days0.append(current)
    else:pass

print(days0)

##Calculates the Normal distribution

import math

#Calculates the mean value(average)
suma=0                                      #Sum of all inputs
sumA=0
countA=0
sumB=0
countB=0
sumAB=0
countAB=0
sum0=0
count0=0


for i in notWorkDays:
    suma=suma+i
    count=count+1
    
mean=suma/count


for i in daysA:
    sumA=sumA+i
    countA=countA+1
    
meanA=sumA/countA

for i in daysB:
    sumB=sumB+i
    countB=countB+1
    
meanB=sumB/countB

for i in daysAB:
    sumAB=sumAB+i
    countAB=countAB+1
    
meanAB=sumAB/countAB

for i in days0:
    sum0=sum0+i
    count0=count0+1
    
mean0=sum0/count0


#Calculates standard deviation

psd=0            #population standard deviation
psdA=0            
psdB=0
psdAB=0
psd0=0

for i in range(0,countA): psd=psd+math.sqrt((notWorkDays[i]-mean)**2/count)    #Combines results of all data

for i in range(0,countA): psdA=psdA+math.sqrt((daysA[i]-meanA)**2/countA)

for i in range(0,countB): psdB=psdB+math.sqrt((daysB[i]-meanB)**2/countB)

for i in range(0,countAB): psdAB=psdAB+math.sqrt((daysAB[i]-meanAB)**2/countAB)

for i in range(0,count0): psd0=psd0+math.sqrt((days0[i]-mean0)**2/count0)

##############################################################################

# Importing required libraries
 
# import required libraries
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
 
# Creating the distribution
data= np.arange(1,50,1)
dataA = np.arange(1,50,1)
dataB = np.arange(1,50,1)
dataAB= np.arange(1,50+1,1)
data0 = np.arange(1,50,1)
pdf = norm.pdf(data , loc= mean , scale = psd )
pdfA = norm.pdf(dataA , loc= meanA , scale = psdA )
pdfB = norm.pdf(dataB , loc= meanB , scale = psdB )
pdfAB = norm.pdf(dataAB , loc= meanAB , scale = psdAB )
pdf0 = norm.pdf(data0 , loc= mean0 , scale = psd0 )
#Visualizing the distribution
 
sb.set_style('whitegrid')
sb.lineplot(x= data, y= pdf*1000 , color = 'orange')
sb.lineplot(x= dataA, y= pdfA*1000 , color = 'black')
sb.lineplot(x=dataB, y=pdfB*1000 , color = 'red')
sb.lineplot(x=dataAB, y=pdfAB*1000 , color = 'green')
sb.lineplot(x=data0, y=pdf0*1000 , color = 'orange')
plt.xlabel('Broj dana bolniske')
plt.ylabel('Procenat ljudi %')
plt.show()
