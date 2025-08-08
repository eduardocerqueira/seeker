#date: 2025-08-08T17:12:37Z
#url: https://api.github.com/gists/79bf8ecaa668c4a369c7c43077afb998
#owner: https://api.github.com/users/datavudeja

#------------
# Import
#------------
#CSV
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', header=0,index_col=0, quotechar='"',sep=':', na_values = ['na', '-', '.', ''])

# series
df = pd.concat([s1, s2], axis=1)

# Python dictionary by column
df = DataFrame({'col0' : [1.0, 2.0, 3.0, 4.0],'col1' : [100, 200, 300, 400]})

# python dictionary by row
df = DataFrame.from_dict({ 'row0' : {'col0':0, 'col1':'A'},'row1' : {'col0':1, 'col1':'B'}}, orient='index')
df = DataFrame.from_dict({ 'row0' : [1, 1+1j, 'A'],'row1' : [2, 2+2j, 'B']}, orient='index')


#HDF5
pd.read_hdf('foo.h5','df')

#-------
# save
#------
df.to_csv('name.csv', encoding='utf-8')
df.to_hdf('foo.h5','df')
