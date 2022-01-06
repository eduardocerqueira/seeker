#date: 2022-01-06T17:18:43Z
#url: https://api.github.com/gists/9442a19432d2835a00b0177b274002df
#owner: https://api.github.com/users/Pinnacle55

def has_player_had_a_bye(player, match_history):
    """
    Looks through match_history to see if the player has had a bye
    
    returns Boolean
    """
    
    for match in match_history:
        if player == match_history[match]["Player_1"]:
            if match_history[match]["Player_2"] == "Bye":
                return True
        elif player == match_history[match]["Player_2"]:
            if match_history[match]["Player_2"] == "Bye":
                return True
            
    return False
    
# If there odd number of players, the worst performing player has a bye
if len(player_list) % 2 != 0:

    # Iterate backwards starting from the worst player
    # This list slice might be a little difficult to understand, but it basically just goes
    # from -1, -2, -3...
    for player in player_list[-1::-1]:

        # We need to check if the worst performing player already had a bye. 
        # In which case they cannot have a bye again.
        if has_player_had_a_bye(player, match_history):
            continue

        # If the player doesn't have a bye, then pop that player from the list
        # and give them a bye. You actually don't need this "else:" statement, but I like it
        # to make things easier to read.
        else:
            pairings.append([player_list.pop(player_list.index(player)), "Bye"])
            # break the for loop
            break
   