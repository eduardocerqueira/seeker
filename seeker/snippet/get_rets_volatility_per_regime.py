#date: 2023-10-11T16:59:13Z
#url: https://api.github.com/gists/17e18a872fed269ad24da9251f846ed2
#owner: https://api.github.com/users/quantra-go-algo

# Find the returns standard deviation for state 0
state0_vol = data['returns'][data['states']==0].std()*np.sqrt(252)*100

# Find the returns standard deviation for state 1
state1_vol = data['returns'][data['states']==1].std()*np.sqrt(252)*100

# Print the returns volatility for both states
print(f'Volatility for state 0 and 1 are {state0_vol:.2f} and {state1_vol:.2f}, respectively')