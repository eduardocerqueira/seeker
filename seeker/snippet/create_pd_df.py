#date: 2023-02-24T16:59:30Z
#url: https://api.github.com/gists/11e00b46013af2fdd57584d427811710
#owner: https://api.github.com/users/StatsGary

df = pd.DataFrame(
    list(zip(test_inputs, risks, list(np.array(preds)))), 
    columns=['Prompts', 'RiskName', 'Label']
    )
print(df)