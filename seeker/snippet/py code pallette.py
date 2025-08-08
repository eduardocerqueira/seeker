#date: 2025-08-08T17:01:43Z
#url: https://api.github.com/gists/eaece25aa33c7248c7c8a89fea284b64
#owner: https://api.github.com/users/datavudeja

# *-----------------------------------------------------------------
# | PROGRAM NAME: py code pallette.py
# | DATE: 7/17/20 (original: 12/28/18)
# | CREATED BY: MATT BOGARD 
# | PROJECT FILE:             
# *----------------------------------------------------------------
# | PURPOSE: examples of data munging and analysis in python
# *----------------------------------------------------------------

# see also: https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_r.html

###############################################################################
###############################################################################
#################          under construction      ############################
###############################################################################
###############################################################################


#----------------------------------
# basic data management
# ---------------------------------

pd.set_option('display.max_rows', None) # to get untruncated output (see: https://thispointer.com/python-pandas-how-to-display-full-dataframe-i-e-print-all-rows-columns-without-truncation/)

pd.set_option('display.float_format', '{:.2f}'.format) # suppress sci notation


# list all data frames
sheets=[]    
for var in dir():
    if isinstance(locals()[var], pd.core.frame.DataFrame)  and var[0]!='_':
        sheets.append(var)
        
print(sheets) 


# delete and release data frame from memory

import gc

del [[df]]
gc.collect()
df = pd.DataFrame()

# multiple dfs

del [[df1,df]]
gc.collect()
df1=pd.DataFrame()
df=pd.DataFrame()

### reading and writing data

import pandas as pd

df1 = pd.read_csv('bank_marketing.csv')

# Create a list of the new column labels: new_labels
new_labels = ['year','ID']

# Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv('bank_marketing.csv', header=0, names=new_labels)

# example with specified file path
df = pd.read_csv('C:/Users/Documents/Tools and References/Data/bank_marketing.csv')

# Save the cleaned up DataFrame to a CSV file without the index
df2.to_csv(file_clean, index=False)

# import by including index_col
cars = pd.read_csv('cars.csv', index_col = 0)

# find working directory
import os 
os.getcwd() 

print("Current Working Directory " , os.getcwd()) # check & print current directory


### export or 'pickle' python data frame
# export data frame for future 

# change working directory
 os.chdir("//Projects//Data")

# export as pickle file
df.to_pickle('df.pkl')

# remove previous copy
import gc

# read file back in - un-pickle
df = pd.read_pickle('df.pkl')

### export to csv with a specified file path vs working directory
bank_mkt_scored.to_csv('test123.csv') # this will write to working directory
                     
path='C:\\Users\\mtb2901\\Documents\\Tools and References\\Data' 
path2 = path + '\\test123.csv'
bank_mkt_scored.to_csv(path2)  # this writes to directory above

# write to text file 
df.to_csv('ProgramOutcome_DE_DB_2.txt', sep='\t', index=False)


### reading a txt file

import pandas as pd

# Assign filename: file
file = 'P:/Data/cohortfile.txt'

# Import file: data
df = pd.read_csv(file, sep='\t',encoding = "ISO-8859-1")


# list all data frames
sheets=[]    
for var in dir():
    if isinstance(locals()[var], pd.core.frame.DataFrame)  and var[0]!='_':
        sheets.append(var)
        
print(sheets)



#
# extracting zip files from web
#

import requests
import zipfile
import io

url = 'https://ihmecovid19storage.blob.core.windows.net/latest/ihme-covid19.zip'
r = requests.get(url,stream=True)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('//econometrics//data')





#### cleaning up temp files

# delete df to clear up memory
del X
del X_score
del X_test
del X_train

### create a toy pandas data frame

# define a dictionary

data = {'GARST' :[150,140,145,137,141,145,149,153,157,161],
        'PIO':[160,150,146,138,142,146,150,154,158,162],
		'MYC':[137,148,151,139,143,120,115,136,130,129],
		'DEK':[150,149,145,140,144,148,152,156,160,164],
        'WYF':[110,125,135,122,127,132,130,'NaN',147,119],
		'PLOT':[1,2,3,4,5,6,7,8,9,10],
		'BT': ['Y','Y',	'N','N','N','N','Y','N','Y','Y'],
		'RR':['Y','N','Y','N','N','N','N','Y','Y','N'],
         'ID':[1,2,3,4,5,6,7,8,9,10]
}

# convert to a data frame
df = pd.DataFrame(data,columns=['ID','GARST','PIO','MYC','DEK','WYF','PLOT','BT','RR'])
print(df)

### inspecting your data

df.columns
list(df.columns) # untruncated list
sorted(list(df2.columns)) # sorted list
df.head()
df.tail()
df.info()
df.shape()
df.index

### misc things you can do to manipulate values and fields

df['ID'] = df['ID'].astype(str) # convert ID to string
df['PLOT'] = df['PLOT'].astype(str) # convert plot to string
df['WYF'] = pd.to_numeric(df['WYF'], errors='coerce', downcast ='integer')

### example working with data types

df['xvar'] = df['GARST']
df['xvar'] = df['xvar'].astype(str) # convert to string

# use pandas to convert back to numeric
df['xvar'] = pd.to_numeric(df['xvar'], errors='coerce')

# pad leading zeros
df['GARST0'] = df['GARST'].apply(lambda x:'{:0>6}'.format(x))

df.head()

# create unique ID for each line in data frame
df['ID2'] =  list(range(1,11))

# Print out type of GARST
print(type(df['GARST']))

# Print out length of var1
print(len(df['GARST']))

# drop variables
df.drop(['xvar', 'ID2'], axis=1)

# create dummy variables (example)
df = pd.get_dummies(df, columns=['ORIGIN', 'DEST'])
df.head()


#----------------------------------
# working with duplicated data
#----------------------------------

data = {'WT' :[150,148,145,200,198,196,191,175,175,161],
        'HT':[72,72,72,68,68,68,68,69,69,71],
		'EX': ['Y','Y','Y','Y','Y','Y','Y','N','N','N'],
		'DIET':['N','N','N','Y','Y','Y','Y','N','N','N'],
         'PLAN':['A','A','B','A','A','B','B','A','A','B'],
         'ID':[1,1,1,2,2,2,2,3,3,4]
}

# convert to a data frame
tmp = pd.DataFrame(data,columns=['ID','WT','EX','DIET','PLAN'])
print(tmp)

# get unique members
tmp2 = tmp.drop_duplicates(['ID'])
print(tmp2)

# get unique id and plan combinations
tmp3 = tmp.drop_duplicates(['ID','PLAN'])
print(tmp3)

# identify duplicates (example)
tmp = df1.groupby(['ID']).size().reset_index(name='count') # count duplicates
tmp = tmp.sort_values(['count'], ascending=[False]) # sort
print(tmp) # check


#-----------------------------------
# high level summary stats
#-----------------------------------

df.describe() # summary stats for all numerical colunms

# example summary stats with specified percentiles for variable yields
perc =[.25, .50, .75, .90,.95,.99]
df1.yields.describe(percentiles = perc)

# Print the value counts for BT and RR
print(df['BT'].value_counts(dropna=False))
print(df['RR'].value_counts(dropna=False))

# sort by GARST yield
df= df.sort_values('GARST', ascending=False)
print(df)

df= df.sort_values('GARST')
print(df)

# sort ascending by trait and by descending GARST yield_data

df = df.sort_values(['BT', 'GARST'], ascending=[True, False])
print(df)

df = df.sort_values(['BT', 'GARST'], ascending=[False, False])
print(df)

df = df.sort_values(['BT', 'GARST'], ascending=[False, True])
print(df)

# sort by index to restore original order
df = df.sort_index()
print(df)


#------------------------------------
# subsetting and basic data frame manipulation
#------------------------------------

# Print out GARST column as Pandas Series
print(df['GARST'])

# Print out GARST column as Pandas DataFrame (use double brackets)
print(df[['GARST']])

# Print out DataFrame with GARST and PLOT columns
print(df[['GARST','PLOT']])

# subset data via variable selection  
my_hybrids = df[['GARST','PIO']]
my_hybrids.head() # check
my_hybrids.info() # check

### example using .loc

# Create the list of column labels: cols and get rows
cols = ['GARST','PIO','PLOT']
rows = df.index

# Create the new DataFrame
my_hybrids= df.loc[rows,cols]
print(my_hybrids)

# get only the first three rows
cols = ['GARST','PIO','PLOT']
rows = [0,1,2]
my_hybrids= df.loc[rows,cols]
print(my_hybrids)

### subset / subquery similar to %in% operator in R (example)

dups = df1[df1[ID'].isin(['99999991',    
                          '99999992',    
                          '99999993',    
                          '99999994',])] 
# 'not' in (example)
df2 = df1[~df1[ID'].isin(['99999991',    
                          '99999992',    
                          '99999993',    
                          '99999994',])] 


### filtering based on logical conditions

# numpy logical operators
# np.logical_and(), np.logical_or() and np.logical_not()

# subset data based on observed values

import numpy as np

# define logical condition
hi = np.logical_and(df['GARST'] > 150, df['PIO'] > 150)
hi_yields = df[hi] # subset df based on condition
hi_yields.head() # check

# define logical condition
stack = np.logical_and(df['BT'] == "Y",df['RR'] == "Y")
stacked_traits = df[stack] # subset based on condition
stacked_traits.head() # check


# we don't have to use numpy
stack = (df['BT'] == "Y") & (df['RR'] == "Y")
stacked_traits = df[stack]
print(stacked_traits)

# or similarly
mask = (tmp['year'] == 2016)
tmp = tmp[mask]

# Create the boolean array: high_turnout
hi = df['GARST'] > 150 

# Filter with array: hi
hi_garst = df.loc[hi]
print(hi_garst)

#------------------------------------------
# if else logic
#------------------------------------------

def traits(BT):
    # retunr gmo vs non-gmo trait
    if BT == "Y":
        return "bt"
    else:
        return "non-bt"
    
df['trait'] = df.BT.apply(traits)    
print(df)

# you can do this with a lambda function

df['trait'] = df['BT'].apply(lambda BT: 'bt' if BT == 'Y' else 'non-bt')
print(df)


def elite(x):
    max_yield = max(x)
    yield_advantage =  max_yield - 150
    return yield_advantage

df_elites = df[['GARST','PIO','DEK','MYC']].apply(elite)
print(df_elites)

### create categories based on value ranges

conditions = [
    (df['GARST'] < 140),
    (df['GARST'] >= 140) & (df['GARST'] < 150),
    (df['GARST'] >= 150)]
choices = ['low', 'med', 'high']
df['lvl'] = np.select(conditions, choices, default='na')
print(df)


# another example
conditions = [
    (df['RR'] == "Y"),
    (df['BT'] == "Y")]
choices = ['gmo', 'gmo']
df['gmo'] = np.select(conditions, choices, default='non-gmo')

# compact for two levels
df['trait'] = np.where(df['BT']=='Y', 'bt', 'non-bt')
print(df)

# create a binary flag
df['flag'] = np.where(df['gmo']=='gmo', 1,0)
print(df)

df.columns

### example from datacamp - creating custom segments
    
# Define rfm_level function
def rfm_level(df):
    if df['RFM_Score'] >= 10:
        return 'Top'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 10)):
        return 'Middle'
    else:
        return 'Low'

# Create a new variable RFM_Level
datamart['RFM_Level'] = datamart.apply(rfm_level, axis=1)


#-------------------------------------
# merging data
#------------------------------------


# inner
pd.merge(bronze, gold, on=['NOC', 'Country'],suffixes=['_bronze', '_gold'], how='inner') 

# left
pd.merge(bronze, gold, on=['NOC', 'Country'],suffixes=['_bronze', '_gold'], how='left') 

# right
pd.merge(bronze, gold, on=['NOC', 'Country'],suffixes=['_bronze', '_gold'], how='right') 

# outer
pd.merge(bronze, gold, on=['NOC', 'Country'],suffixes=['_bronze', '_gold'], how='outer') 

# other examples

df3 = df1.merge(df2[['ID','Date','Country']], on = ['ID','Date], how = 'left')




#------------------------------------
# strings
#------------------------------------

### string methods

# ex
variety = "garst8590bt"
print(type(variety)) 

# convert to upcase
variety_up = variety.upper()

print(variety)
print(variety_up)

# Print out the number of t's in variety
print(variety.count('t'))

# strip white spaces
df.columns = df.columns.str.strip()

string = "freeCodeCamp"
print(string[0:5])

# substring operations
data = {'HYBRID' :['P1324RR','P1498RR','P1365BT','DKC2988RR','DKC4195BT'],
        'YIELD':[160,150,146,138,142],
         'ID':[1,2,3,4,5]
}

# convert to a data frame
df = pd.DataFrame(data,columns=['ID','HYBRID','ID'])
print(df)

df['HYBRID_CD'] = df.HYBRID.str.slice(0, 1) # this gets the first character
df['HYBRID_CD2'] = df.HYBRID.str[:1] # this gets the first character
df['HYBRID_CD2'] = df.HYBRID.str[1:] # this skipps the first character
df['TRIAT'] = df['HYBRID'].str[-2:] # this gets the last 2 characters

#------------------------------------
# loops
#-----------------------------------

### loop over a list
    
# ex

garst= [150,140,145,137,141]
for yields in garst : 
    print(yields)
    
###  basic loop
x = 1
while x < 4 :
    print(x)
    x = x + 1

### loop over data frame

# ex: create new string column for BT that is lower case
for lab, row in df.iterrows() :
    df.loc[lab,"bt"] = row["BT"].lower()
    
df.head() # check

# this can similarly be accomplished via 'apply' with string function
df["rr"] = df["RR"].apply(str.lower)
df.head() # check

# ex create new column giving amount of GARST yielding > 150 (or under)
for lab, row in df.iterrows() :
    df.loc[lab, "amt_over_150"] = (row["GARST"] - 150)

df.head() # check


#---------------------------------------------
# functions
#---------------------------------------------

# example of function syntax
def fun(a,b):
    """State what function does here"""
    # Computation performed here
    return x, y


# define function
def square(num):
    new_value = num ** 2
    return new_value

square(5) # call function

### basic function to calculate amount over 150 for a given hybrid
def diff(df,hybrid):
    "calculates difference in yield from 150"
    for lab, row in df.iterrows() :
     df.loc[lab, "amt_over_150"] = (row[hybrid] - 150)
     

diff(df,"PIO") # call function
df.head() # check


### function that finds the yeild advantage for a variety

def elite(df,var1):
    max_yield = max(df[var1])
    yield_advantage =  max_yield - 150
    return yield_advantage

elite(df,"GARST") # for garst
elite(df,"PIO") # for pio
elite(df,"DEK") # for dekalb
elite(df,"MYC") # for myc

# make the function more general and use apply

def elite(x):
    max_yield = max(x)
    yield_advantage =  max_yield - 150
    return yield_advantage

df_elites = df[['GARST','PIO','DEK','MYC']].apply(elite)
print(df_elites)

# application: determine which hybrid has
# at least a 10 bu yield advantage



#---------------------------------------------
# transposing or reshaping data
#---------------------------------------------

# use melt to convert columns into rows
df2 = pd.melt(df, id_vars=['ID','BT','RR'], var_name='hybrid', value_name='yield')

print(df2) # check

df2['yield'] = df2['yield'].astype(int) # fix loss of format

# pivot this data back into tidy form
df3 =  df2.pivot_table(index = ['ID','BT','RR'],columns = 'hybrid',values= 'yield')
df3.head() # check

# convert indices back to columns
df3.reset_index(level=['ID','BT','RR'])

df3.reset_index() # this would have worked too

df3.info()

### example with toy panel data

# create toy panel (long) data
data = {'id' :[1,1,1,2,2,2,3,3,3],
        'measure':["depth","temp","width","depth","temp","width","depth","temp","width"],
		'values': [2,50,18,1.5,53,18,2.5,60,18],

}

# convert to a data frame
tmp = pd.DataFrame(data,columns=['id','measure','values'])
print(tmp)


# pivot this data back into tidy form (wide)
df1 =  tmp.pivot_table(index = ['id'],columns = 'measure',values= 'values')

# convert indices back to columns
df1 = df1.reset_index()

print(df1) # check

# use melt to convert columns back into rows (panel or long)                
df2 = pd.melt(df1, id_vars=['id'], var_name='measure', value_name='values')
print(df2)


#--------------------------------------
# group by operations
#--------------------------------------

df.groupby('BT')['GARST'].mean() # average yield for garst BT

# check this the hard way:
garst = df[['GARST','BT']] # double brackets gives a data frame vs series in python
print(garst)

bt = garst['BT'] == "Y" 
print(garst.loc[bt]) # check
print(garst.loc[~bt]) # check

garst['GARST'].loc[bt].mean() # average for BT
garst['GARST'].loc[~bt].mean() # average for non-BT

# group by list of hybrids of interest
hybrids = ['GARST','PIO','MYC']
df.groupby('BT')[hybrids].mean()

# overall means by BT trait
df.groupby('BT').mean()

# check the hard way
df.loc[df['BT'] == 'Y'].mean() # just get mean for BT overall
df.loc[df['BT'] == 'N'].mean() # just get mean for non BT overall

# create new data frame aggregating by variable
tmp2 = tmp.groupby('ID')[['purchases']].mean().reset_index() 

### aggregations using agg functon

# see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html

import numpy as np

# create toy data (note how this leverages numpy to create NaN values)
df = pd.DataFrame([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [np.nan, np.nan, np.nan]],
                   columns=['A', 'B', 'C'])

# ex: min and max for each column
df.agg(['sum', 'min'])

# different aggregations specific for each specified column
df.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})

# aggregate over columns
df.agg("mean", axis="columns")

# use aggregate to create new variables and custom function
data_mart = df.agg({
    'A': lambda x: ( x.max() - x),
    'B': 'count',
    'C': 'mean',
     })

# Rename the columns 
data_mart.rename(columns={'A': 'MaxDiff',
                         'B': 'Frequency',
                         'C': 'Mean'}, inplace=True)


# apply to panel data

# create toy panel (long) data
data = {'id' :[1,1,1,2,2,2,3,3,3],
        'measure':["depth","temp","width","depth","temp","width","depth","temp","width"],
		'values': [2,50,18,1.5,53,18,2.5,60,18],

}

# convert to a data frame
df = pd.DataFrame(data,columns=['id','measure','values'])
print(df)

# sum total values by id
data_mart = df.groupby(['id']).agg({'values': 'sum'})

#-----------------------------------------
# cross tabs
#----------------------------------------

# for more info: https://pbpython.com/pandas-crosstab.html 

pd.crosstab(dat.BT, dat.RR) 

pd.crosstab(df1.Gender_Code,df1.treat, normalize=True) # gives % in each combination
pd.crosstab(df1.treat,df1.Diag_Grouping, normalize='index') # example: normalize gives % row wise
pd.crosstab(df1.Diag_Grouping,df1.treat, normalize='columns') # this is easier to read columnwise


# see also groupby

#--------------------------------------
# missing data
#--------------------------------------

df.isnull().sum(axis=0) # count of missing values by variable in dataframe

df1.isnull().sum() # total missing per column
						    
df[df.isnull().values.any(axis=1)].head() # check data where there are missing values

tmp = df.isnull().sum(axis=0) # check missing values
df.dropna(subset=['ZipCode','Income'], how='all', inplace=True) # drop variables with missing values

# drop missing values
wyf = df[["WYF"]]
wyf2 = wyf.dropna(axis=0,how='any')

# how does python handle missing data
wyf.mean()
wyf2.mean()

wyf.std()
wyf2.std()


del wyf
del wyf2

# Replace the NaN price values with 0 
purchase_data.price = np.where(np.isnan(purchase_data.price),0, purchase_data.price)

# replace missing categorical

conditions = [(tmp['RiskScore'].isnull() == True)]
choices = ['Missing']
tmp['HasRiskScore'] = np.select(conditions, choices, default='Yes')
print(tmp['HasRiskScore'].value_counts(dropna=False)) # check values

### replace misisng and none string values

# create data with blank and None string values
data = {'GARST' :[150,140,145,137,141,145,149,153,157,161],
        'PIO':[160,150,146,138,142,146,150,154,158,162],
		'MYC':[137,148,151,139,143,120,115,136,130,129],
		'DEK':[150,149,145,140,144,148,152,156,160,164],
        'WYF':[110,125,135,122,127,132,130,'NaN',147,119],
		'PLOT':[1,2,3,4,5,6,7,8,9,10],
		'BT': ['Y','Y',	None,'N','N','N','Y','N','Y',''],
		'RR':['Y','N','Y','N','N','N','N','Y',None,'N'],
         'ID':[1,2,3,4,5,6,7,8,9,10]
}

# convert to a data frame
df = pd.DataFrame(data,columns=['ID','GARST','PIO','MYC','DEK','WYF','PLOT','BT','RR'])
print(df)

# preview data
df.head(10)

# replace None with 'MISSING' 
df = df.replace({None: 'Missing'})

# replace blank with missing
df = df.replace({'': 'Missing'})

#-------------------------------------
# imputation
#-------------------------------------

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

df.WYF.median() # waht is the median

# Impute median
df['WYFimp'] = df.WYF.transform(impute_median)
print(df) # check


#--------------------------------------
# basic descriptives
#--------------------------------------

print(df['RR'].value_counts(dropna=False))

df.GARST.mean()
df.GARST.std()
df.GARST.var()

#-------------------------------------
# covariance matrix operations
#-------------------------------------

covariance_matrix = np.cov(df.GARST,df.PIO)

# Print covariance matrix
print(covariance_matrix)

# Extract covariance of garst and pioneer
garst_pio= covariance_matrix[0,1]
print(garst_pio)

# same as above
garst_pio= covariance_matrix[1,0]
print(garst_pio)

# variance of pioneer
pio_var= covariance_matrix[1,1]
print(pio_var)

# variance of garst
garst_var= covariance_matrix[0,0]
print(garst_var)

#-------------------------------------
# data visualizatiion
#------------------------------------

# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### boxplot

# Create the boxplot
df.boxplot(column = 'GARST')
df.boxplot(column='GARST', by='RR', rot=90)

# Display the plot
plt.show()


### Box-and-whisker plot with seaborn

# Create box plot with Seaborn's default settings
_ = sns.boxplot(x='RR', y='GARST', data=df)

# Label the axes
_ = plt.xlabel('RR')
_ = plt.ylabel('yield')

# Show the plot
plt.show()

### countplot /  bar plot
sns.countplot(x="total_flags", data=df2)
						    
speed = [0.1, 17.5, 40, 48, 52, 69, 88]
lifespan = [2, 8, 70, 1.5, 25, 12, 28]
index = ['snail', 'pig', 'elephant', 'rabbit', 'giraffe', 'coyote', 'horse']
df = pd.DataFrame({'speed': speed,'lifespan': lifespan}, index=index)

print(df)
						    
ax = df.plot.bar(rot=0)
						    
ax = df.plot.bar(y='speed', rot=0) # single column
						    					   

### histogram

# Plot histogram of versicolor petal lengths
_ = plt.hist(df.GARST)

# Label axes
_ = plt.xlabel('yield')
_ = plt.ylabel('count')

# Show histogram
plt.show()



### histogram with pandas

# Plot the PDF
df.GARST.plot( kind='hist', normed=True, bins=30, range=(100,200))
plt.show()

# Plot the CDF
df.GARST.plot( kind='hist', normed=True, cumulative = True, bins=30, range=(100,200))
plt.show()

### bee swarm plot

_ = sns.swarmplot(x='RR', y='GARST', data=df)
_ = plt.xlabel('RR')
_ = plt.ylabel('yield')
plt.show()


### ECDF 
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1,n+1) / n

    return x, y

### Plotting the ECDF
    
# Compute ECDF for versicolor data: x_vers, y_vers
x_val, y_val = ecdf(df.GARST)

# Generate plot
_ = plt.plot(x_val, y_val, marker='.',linestyle = 'none')

# Label the axes
_ = plt.xlabel('yield')
_ = plt.ylabel('ECDF')


# Display the plot
plt.show()


### Computing percentiles

# Specify array of percentiles: percentiles
percentiles = [2.5,25,50,75,97.5]

# Compute percentiles: ptiles_vers
ptiles = np.percentile(df.GARST, percentiles)

# Print the result
print(ptiles)

### scatter plot

_ = plt.plot(df.GARST, df.PIO, marker='.', linestyle='none')


# Label the axes
_ = plt.xlabel('garst')
_ = plt.ylabel('pioneer')

# Show the result
plt.show()




#----------------------------------------
# time series
#----------------------------------------

### create data frame with date time index

data = {'date': ['2014-05-01 18:00:00', '2014-05-01 18:30:00', '2014-05-02 17:00:00', '2014-05-02 16:00:00', '2014-05-02 15:30:00', '2014-05-02 14:00:00', '2014-05-03 13:00:00', '2014-05-03 18:00:00', '2014-04-30 15:00:00', '2014-04-30 18:00:00'], 
        'aphids': [15, 25, 26, 12, 17, 14, 26, 32, 48, 41]}
df = pd.DataFrame(data, columns = ['date', 'aphids'])
print(df)

df.info()

# Convert df['date'] from string to datetime

# Prepare a format string: time_format
time_format = '%Y-%m-%d %H:%M'

# Convert date_list into a datetime object: my_datetimes
df['date'] = pd.to_datetime(df['date'], format=time_format)  

df.info()

# function for parsing calendar date
import datetime as dt

def get_day(x): return dt.datetime(x.year, x.month, x.day) 

# Create day column
df['day'] = df['date'].apply(get_day) 
df.head()

# Set df['date'] as the index and delete the column
df.index = df['date']
del df['date']
df # check
df.info() # check

### date operations

# Extract the hour from 2pm to 4pm on '2014-05-02': ts1
ts1 = df.loc['2014-05-02 14:00:00':'2014-05-02 16:00:00']

# Extract '2014-05-02': ts2
ts2 = df.loc['2014-05-02']

# Extract data from '2014-05-03' to '22014-05-02': ts3
ts3 = df.loc['2014-05-03':'2014-05-05']

# Downsample to get total within 2 hours 
df1 = df['aphids'].resample('2h').sum()
print(df) # compare
print(df1) # check

# Downsample to get daily total aphid counts
df1 = df['aphids'].resample('D').sum()

# get daily high counts
daily_highs = df['aphids'].resample('D').max()

# get counts for april
april = df['aphids']['2014-Apr']
print(april)

# examples from DataCamp Customer Analytics and A/B Testing

# Provide the correct format for the date  Saturday January 27, 2017
date_data_one = pd.to_datetime(date_data_one, format="%A %B %d, %Y")
print(date_data_one)

# Provide the correct format for the date 2017-08-01
date_data_two = pd.to_datetime(date_data_two, format= "%Y-%m-%d")
print(date_data_two)

# Provide the correct format for the date 08/17/1978
date_data_three = pd.to_datetime(date_data_three, format='%m/%d/%Y')
print(date_data_three)

# Provide the correct format for the date 2016 March 01 01:56
date_data_four = pd.to_datetime(date_data_four, format="%Y %B %d %H:%M")
print(date_data_four)

# great reference: https://www.dataquest.io/blog/python-datetime-tutorial/ 

# hacky way to deal with dirty stringy dates
df3['END_DT'] = df3['END_DT'].str.slice(stop=10)
df3['END_DT'] = pd.to_datetime(df3.END_DT, format='%Y/%m/%d')
print(df3.END_DT.head())


# calculate 12 months pre and post dates based on index date 

df_chrt['date_post'] = df_chrt['INDEX_DT'] + pd.DateOffset(months=12)
df_chrt['date_pre'] = df_chrt['INDEX_DT'] - pd.DateOffset(months=12)

#-------------------------------------
# random numbers
#-------------------------------------

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

### random uniform 

# Seed the random number generator
np.random.seed(42) 

# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)

# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()

# Plot a histogram
_ = plt.hist(random_numbers)

# Show the plot
plt.show()

# not sure why a loop is always necessary
rs = np.random.random(10000)
plt.hist(rs)

# both approahces create an array containing the random values
type(rs)
type(random_numbers)

#### poisson

samples_poisson = np.random.poisson(3,size =10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

plt.hist(samples_poisson)

### normal distribution

normdist = np.random.normal(20, 1, size=100000) 

# pdf - histogram
plt.hist(normdist ,bins=100,normed=True,histtype='step')

# normal cdf

x1, y1 = ecdf(normdist)
_ = plt.plot(x1,y1, marker='.',linestyle = 'none')
_ = plt.legend(('normal cdf'), loc='lower right')
plt.show()



### compare GARST to a normal distribution

mean = np.mean(df.GARST) # get empirical mean
std = np.std(df.GARST)  # get empirical std deviation
rnorm = np.random.normal(mean, std, size=1000)  # simulate normal data
x, y = ecdf(df.GARST) # empirical distribution
x_norm, y_norm = ecdf(rnorm)  # normal distribution

sns.set() # apply seaborn templates

_ = plt.plot(x_norm, y_norm)  # plot normally distributed data points
_ = plt.plot(x, y, marker='.', linestyle='none') # plot empirical data
_ = plt.xlabel('yield') 
_ = plt.ylabel('CDF') 
plt.show() 
						    
#------------------------------------------------------
# simulation
#------------------------------------------------------
						    
df1['prob'] = np.random.uniform(0, 1, df1.shape[0]) # add random uniformly distributed value as a column						    
