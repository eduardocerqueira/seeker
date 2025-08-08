#date: 2025-08-08T17:12:37Z
#url: https://api.github.com/gists/79bf8ecaa668c4a369c7c43077afb998
#owner: https://api.github.com/users/datavudeja


# creation of series
s = pd.Series([1,3,5,np.nan,6,8])

# dataframe from series
df = pd.Dataframe(s, columns=['supercolname'])

# creation with index, numpy data, and column name
dates = pd.date_range('20130101',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))

# create from data without index
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]
BabyDataSet = list(zip(names,births))
df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])


# creation from dictionary
In [10]: df2 = pd.DataFrame({ 'A' : 1.,
                              'B' : pd.Timestamp('20130102'),
                              'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                              'D' : np.array([3] * 4,dtype='int32'),
                              'E' : pd.Categorical(["test","train","test","train"]),
                              'F' : 'foo' }

# output
#    A          B  C  D      E    F
# 0  1 2013-01-02  1  3   test  foo
# 1  1 2013-01-02  1  3  train  foo
# 2  1 2013-01-02  1  3   test  foo
# 3  1 2013-01-02  1  3  train  foo
