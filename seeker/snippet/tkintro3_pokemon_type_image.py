#date: 2022-07-30T19:07:41Z
#url: https://api.github.com/gists/4d136a487a20db4423ae72e1dcf19f3f
#owner: https://api.github.com/users/Pinnacle55

def select_pokemon(move = None):
    '''
    When run, the app should update all fields with the appropriate values based on the 
    pokedex above.
    
    Note that this function is specifically designed for the dictionary defined above - your
    function may be different depending on your data source.
    '''
       
    "..."
    
    ##### Fill in Picture-based information #####
    
    "..."
   
    # Get and display the appropriate image for the pokemon's type
    type1label["image"] = type_dict[pokedex[pokedex.name == selected]["type1"].values[0]]
    
    # Note that many Pokemon have a "None" second typing. In such cases, we turn off the type2label
    # using grid_forget
    if pokedex[pokedex.name == selected]["type2"].values[0] == "none":
        type2label.grid_forget()
    # If it does have a second type, we retrieve it like we did the first one.
    else:
        type2label["image"] = type_dict[pokedex[pokedex.name == selected]["type2"].values[0]]
        type2label.grid(row = 0, column = 1)