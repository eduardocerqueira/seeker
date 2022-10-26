#date: 2022-10-26T17:06:57Z
#url: https://api.github.com/gists/e064038c814280025fc08c513787c6d0
#owner: https://api.github.com/users/AnalyticsForPleasure

def number_of_games_a_player_played_over_his_career(df):
    #global player_name
    player_name_list = []
    avg_games_in_a_season_list = []
    number_of_seasons_played_list = []
    number_of_games_in_a_career = []
    percentage_of_num_of_games_a_season_list = []
    avg_points_in_a_season_list = []

    grouping_by_player_name = df.groupby('player_name')
    for player_name, player_name_df in grouping_by_player_name:
        if all((player_name_df.shape[0] > 8) & (player_name_df['pts'] > 15)):
            print(player_name)
            print(player_name_df.shape[0])
            number_of_seasons = player_name_df.shape[0]
            print(player_name_df['gp'].sum())
            Avg_points_scored_in_a_season = player_name_df['pts'].mean()
            Number_of_games_played_in_a_player_career = player_name_df['gp'].sum()
            avg_games_played_in_each_season = player_name_df['gp'].mean()
            percentage_games_played_a_season = avg_games_played_in_each_season / 83  # Each season 83 games
            print('*')

            player_name_list.append(player_name)
            number_of_games_in_a_career.append(Number_of_games_played_in_a_player_career)
            number_of_seasons_played_list.append(number_of_seasons)
            avg_games_in_a_season_list.append(avg_games_played_in_each_season)
            percentage_of_num_of_games_a_season_list.append(percentage_games_played_a_season)
            avg_points_in_a_season_list.append(Avg_points_scored_in_a_season)

    df_starting = {'Player_name': player_name_list,
                   'Number_of_games_over_the_player_career': number_of_games_in_a_career,
                   'Number of seasons played': number_of_seasons_played_list,
                   'Avg games played in a season': avg_games_in_a_season_list,
                   'Number of games played in a season ( % )':percentage_of_num_of_games_a_season_list,
                   'Avg_points_in_a_season':avg_points_in_a_season_list
                   }

    final_table = pd.DataFrame(df_starting, columns=['Player_name', 'Number_of_games_over_the_player_career',
                                                     'Number of seasons played', 'Avg games played in a season','Number of games played in a season ( % )','Avg_points_in_a_season'])
    final_table.sort_values(by='Avg games played in a season', inplace=True, ascending=False)
    final_table['Number of games played in a season ( % )'] = (final_table['Number of games played in a season ( % )']).apply(lambda r:"{:.2%}".format(r))
    print('*')
    return final_table