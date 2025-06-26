#date: 2025-06-26T16:59:10Z
#url: https://api.github.com/gists/3d472bd51fc942a5eeb57be63078362f
#owner: https://api.github.com/users/LSzubelak

model1 = smf.ols('productivity ~ motivation + training_hours', data=data1).fit()
print(model1.summary())