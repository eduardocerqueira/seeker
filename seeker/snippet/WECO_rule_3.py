#date: 2022-05-10T17:02:55Z
#url: https://api.github.com/gists/b136599a99c9b56a573aaa6c34414741
#owner: https://api.github.com/users/rsalaza4

# Rule 3 - Lower
for i in range(5, len(df_grouped['x_bar'])+1): 
    if((df_grouped['x_bar'][i-4] < df_grouped['-1s'][i-4] and df_grouped['x_bar'][i-3] < df_grouped['-1s'][i-3] and df_grouped['x_bar'][i-2] < df_grouped['-1s'][i-2] and df_grouped['x_bar'][i-1] < df_grouped['-1s'][i-1]) or
       (df_grouped['x_bar'][i-4] < df_grouped['-1s'][i-4] and df_grouped['x_bar'][i-3] < df_grouped['-1s'][i-3] and df_grouped['x_bar'][i-2] < df_grouped['-1s'][i-2] and df_grouped['x_bar'][i] < df_grouped['-1s'][i]) or
       (df_grouped['x_bar'][i-4] < df_grouped['-1s'][i-4] and df_grouped['x_bar'][i-2] < df_grouped['-1s'][i-2] and df_grouped['x_bar'][i-1] < df_grouped['-1s'][i-1] and df_grouped['x_bar'][i] < df_grouped['-1s'][i]) or
       (df_grouped['x_bar'][i-4] < df_grouped['-1s'][i-4] and df_grouped['x_bar'][i-3] < df_grouped['-1s'][i-3] and df_grouped['x_bar'][i-1] < df_grouped['-1s'][i-1] and df_grouped['x_bar'][i] < df_grouped['-1s'][i]) or
       (df_grouped['x_bar'][i-3] < df_grouped['-1s'][i-3] and df_grouped['x_bar'][i-2] < df_grouped['-1s'][i-2] and df_grouped['x_bar'][i-1] < df_grouped['-1s'][i-1] and df_grouped['x_bar'][i] < df_grouped['-1s'][i])):
        R3_lower.append(False)
    else:
        R3_lower.append(True)
    i+=1
    
# Rule 3 - Upper
for i in range(5, len(df_grouped['x_bar'])+1): 
    if((df_grouped['x_bar'][i-4] > df_grouped['+1s'][i-4] and df_grouped['x_bar'][i-3] > df_grouped['+1s'][i-3] and df_grouped['x_bar'][i-2] > df_grouped['+1s'][i-2] and df_grouped['x_bar'][i-1] > df_grouped['+1s'][i-1]) or
       (df_grouped['x_bar'][i-4] > df_grouped['+1s'][i-4] and df_grouped['x_bar'][i-3] > df_grouped['+1s'][i-3] and df_grouped['x_bar'][i-2] > df_grouped['+1s'][i-2] and df_grouped['x_bar'][i] > df_grouped['+1s'][i]) or
       (df_grouped['x_bar'][i-4] > df_grouped['+1s'][i-4] and df_grouped['x_bar'][i-2] > df_grouped['+1s'][i-2] and df_grouped['x_bar'][i-1] > df_grouped['+1s'][i-1] and df_grouped['x_bar'][i] > df_grouped['+1s'][i]) or
       (df_grouped['x_bar'][i-4] > df_grouped['+1s'][i-4] and df_grouped['x_bar'][i-3] > df_grouped['+1s'][i-3] and df_grouped['x_bar'][i-1] > df_grouped['+1s'][i-1] and df_grouped['x_bar'][i] > df_grouped['+1s'][i]) or
       (df_grouped['x_bar'][i-3] > df_grouped['+1s'][i-3] and df_grouped['x_bar'][i-2] > df_grouped['+1s'][i-2] and df_grouped['x_bar'][i-1] > df_grouped['+1s'][i-1] and df_grouped['x_bar'][i] > df_grouped['+1s'][i])):
        R3_upper.append(False)
    else:
        R3_upper.append(True)
    i+=1