#date: 2022-04-15T17:08:09Z
#url: https://api.github.com/gists/2663e6adc4041311fb118dce43da62c9
#owner: https://api.github.com/users/victorfrutuoso

player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 680757, 'full_name'] = 'Steven Kwan'
player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 680757, 'player_age'] = 25

player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 665161, 'full_name'] = 'Jeremy Pena'
player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 665161, 'player_age'] = 25

player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 677951, 'full_name'] = 'Bobby Witt Jr.'
player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 677951, 'player_age'] = 22

player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 679529, 'full_name'] = 'Spencer Torkelson'
player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 679529, 'player_age'] = 23

player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 677594, 'full_name'] = 'Julio Rodriguez'
player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 677594, 'player_age'] = 22

player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 669309, 'full_name'] = 'Joe Perez'
player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 669309, 'player_age'] = 23

player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 681526, 'full_name'] = 'Bryan Lavastida'
player_age_lookup_merge_AL.loc[player_age_lookup_merge_AL['player_id'] == 681526, 'player_age'] = 24

player_age_lookup_merge_AL_final = player_age_lookup_merge_AL[['full_name','player_id','year','player_age']]
player_age_lookup_merge_AL_final['league'] = 'AL'
player_age_lookup_merge_AL_final