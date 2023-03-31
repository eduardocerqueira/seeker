#date: 2023-03-31T16:46:57Z
#url: https://api.github.com/gists/0157b7ea19229c9b90119ec8d6977647
#owner: https://api.github.com/users/Riyasharma-in

# re-order the bounding boxes by their position
for i in range(len(bboxes)-1):
    for ind in range(len(df)-1):
        if df.iloc[ind][4] > df.iloc[ind+1][0] and df.iloc[ind][1] > df.iloc[ind+1][1]:
            #print(df.iloc[ind][4] , df.iloc[ind+1][0] , df.iloc[ind][1] , df.iloc[ind+1][1])
            df.iloc[ind], df.iloc[ind+1] = df.iloc[ind+1].copy(), df.iloc[ind].copy()
            #print(df.iloc[ind], df.iloc[ind+1],'')