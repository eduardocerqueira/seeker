#date: 2023-04-12T16:55:20Z
#url: https://api.github.com/gists/a5be02d6da4b66ce62d65d9b5ae27255
#owner: https://api.github.com/users/srang992

# filtering the taste of Kokoa Kamili and separating the values
taste_encode = choco_data[choco_data['bar_name'].isin(['Kokoa Kamili'])]['taste'].str.get_dummies(sep=', ') 

# fixing some of the values whose pronunciation is wrong
taste_encode['nuts'] = taste_encode['nut'] + taste_encode['nuts'] 
taste_encode['rich_cocoa'] = taste_encode['rich'] + taste_encode['rich cocoa'] + taste_encode['rich cooa']

# dropping the columns containing wrong pronunciation
taste_encode.drop(['nut', 'rich', 'rich cooa'], axis=1, inplace=True)