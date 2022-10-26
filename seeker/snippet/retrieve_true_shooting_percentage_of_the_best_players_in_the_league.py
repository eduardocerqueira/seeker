#date: 2022-10-26T17:09:30Z
#url: https://api.github.com/gists/8c8acb0f5aa94df5e97e8de1e0620044
#owner: https://api.github.com/users/AnalyticsForPleasure

def retrieve_true_shooting_percentage_of_the_best_players_in_the_league(df):
    res = df.loc[:, ['player_name', 'ts_pct', 'season_year', 'gp','pts']]
    res.sort_values(by='ts_pct', ascending=False)

    result = res.loc[(res['gp'] > 50)& (res['pts'] > 10), ['player_name', 'ts_pct', 'season_year', 'gp','pts'],]
    final_result=result.sort_values(by='ts_pct', ascending=False).head(n= 100)
    print('*')
    
    return final_result