#date: 2022-01-04T16:58:10Z
#url: https://api.github.com/gists/073fc660ac9758aa31426793fac1d38c
#owner: https://api.github.com/users/Pinnacle55

def update_standings(match_history,
                     list_of_players):
    """
    Generates the standings of a given match history dictionary.
    
    Requires match_history plus the list of remaining players (in case of drops)
        Also because its easier.
    
    The returned standings_df has the following:
        "Player", "Points", "OMW"
    """
    
    # We have an opponents key, but we'll get rid of it after calculating OMW
    standings_dict = {
        "Player": list_of_players,
        "Played": [],
        "Points": [],
        # Add lists for Opponent Match Win% and Opponents
        "OMW": [],
        "Opponents": []
    }
    
    # For each player 
    for player in list_of_players:
        
        played = 0
        points = 0
        # Create a list that will house the opponents they've played.
        opponents = []
        
        # Check to see if they were involved in a match
        for match in match_history:
            # If they were player 1
            if player == match_history[match]["Player_1"]:
                # Increased played by 1
                played += 1
                # Fill the opponents list with the names of each opponent
                opponents.append(match_history[match]["Player_2"])
                
                # If they won, add 1 point
                if match_history[match]["Result"] == "W":
                    points += 1
                # If they drew, add 0.5 point
                elif match_history[match]["Result"] == "D":
                    points += 0.5
            
            # If they were player 2
            elif player == match_history[match]["Player_2"]:
                
                # Increase games played and add opponent.
                played += 1
                opponents.append(match_history[match]["Player_1"])
                
                # If they drew, add 0.5 point
                if match_history[match]["Result"] == "D":
                    points += 0.5
                    
        # Once we've looped through all the matches
        # Append the stats to the standings_dict
        standings_dict["Played"].append(played)
        standings_dict["Points"].append(points)
        standings_dict["Opponents"].append(opponents)
    
    # We can only calculate OMW after all matches have been added to the standings
    for opponent_list in standings_dict["Opponents"]:
        
        # Create a list with the OMW of each of their opponents.
        running_omw = []
        
        # For each opponent they've previously played, find their win percentage.
        for opponent in opponent_list:
            
            # Find index of their opponent
            opponent_index = standings_dict["Player"].index(opponent)

            # Calculate their OMW by dividing the points they've earned by the 
            # number of games they've played.
            running_omw.append(standings_dict["Points"][opponent_index] / 
                               standings_dict["Played"][opponent_index])

        # If it's the first round, no one has played anyone yet and the length of running_omw 
        # will be 0. 
        if len(running_omw) == 0:
            standings_dict["OMW"].append(0)
        else:
            # Get the average OMW and round to three decimal places. 
            standings_dict["OMW"].append(np.round(np.mean(running_omw), 3))
    
    # Remove the opponents key:value pair
    standings_dict.pop("Opponents")
    
    # Turn the dictionary into a dataframe
    standings_df = pd.DataFrame.from_dict(standings_dict).sort_values(["Points", "OMW"], ascending = False)
    
    return standings_df