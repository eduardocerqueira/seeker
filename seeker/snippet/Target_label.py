#date: 2022-04-15T16:47:57Z
#url: https://api.github.com/gists/7408c89d05b5490001d0404d46983645
#owner: https://api.github.com/users/CIV2T1PEY

merged_df['Default'] = np.where(merged_df['MIS_Status'] == 'CHGOFF', 1, 0)
merged_df['Default'].value_counts()