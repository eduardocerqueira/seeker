#date: 2021-11-30T17:06:28Z
#url: https://api.github.com/gists/7d86ec2b0059f0c9419bc4fa4602082b
#owner: https://api.github.com/users/rahulremanan

df = pd.read_csv(TRAIN_CSV)
display(df.cell_type.unique())
display(df.cell_type.hist())