#date: 2021-09-28T16:58:03Z
#url: https://api.github.com/gists/aba34c5d77d53a712f12104fe464d661
#owner: https://api.github.com/users/woutervanheeswijk

# Set to '%matplotlib widget' to create interactive plot (default is %matplotlib inline)
%matplotlib widget

fig, ax = plt.subplots(figsize=(15, 8)) 

# Add annotations
...

# Set labels
plt.title('Stock price {}'.format(stock_symbol), fontdict = {'fontsize' : 15})
plt.xlabel('Date', fontdict = {'fontsize' : 15})
plt.ylabel('Adj. close price', fontdict = {'fontsize' : 15})

# Plot annotated price series
plt.plot(stock_data['Adj Close'])
plt.show()