#date: 2023-10-11T16:57:14Z
#url: https://api.github.com/gists/afee95a422561cc6b2cf5696a571cdec
#owner: https://api.github.com/users/quantra-go-algo

# Find the R mean for state 0
state0_R_vol = data['R'][data['states']==0].mean()

# Find the R mean for state 1
state1_R_vol = data['R'][data['states']==1].mean()

# Print the R volatility for both states
print(f'Volatility for state 0, 1 and 2 are {state0_R_vol:.2f} and {state1_R_vol:.2f}, respectively')