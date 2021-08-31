#date: 2021-08-31T03:07:45Z
#url: https://api.github.com/gists/6ceda3fd9855472ec2e28b0f74d1ac93
#owner: https://api.github.com/users/gbrfilipe

df3 = pd.concat([df,df2])
df3.drop_duplicates("id",keep=False, inplace = True)
df3.to_csv(full_output_path, sep=",", index=False, header=True)