#date: 2022-07-06T17:13:55Z
#url: https://api.github.com/gists/8c7ed76672b0eb01ed1cedea6b382b7e
#owner: https://api.github.com/users/Pinnacle55

pokedex = {
    "bulbasaur": ["bulbasaur", "grass", "poison", 15, 15, "Grass Starter", 3.8],
    "squirtle": ["squirtle", "water", "none", 10, 10, "Water Starter", 5.9],
    "charmander": ["charmander", "fire", "none", 25, 25, "Fire Starter", 7.3]
}

def select_pokemon():
    '''
    When run, the app should update all fields with the appropriate values based on the 
    pokedex above.
    
    Note that this function is specifically designed for the dictionary defined above - your
    function may be different depending on your data source.
    '''
    # Get the text typed into the Entry widget
    selected = submit_entry.get()
    
    # Update the text variable of the relevant labels with the appropriate information
    # The name
    name_label["text"] = pokedex[selected][0]
    
    # The types
    type1label["text"] = pokedex[selected][1]
    type2label["text"] = pokedex[selected][2]
    
    # the height, weight, species, and catch rate
    height_entry["text"] = pokedex[selected][3]
    weight_entry["text"] = pokedex[selected][4]
    species_entry["text"] = pokedex[selected][5]
    catch_entry["text"] = str(pokedex[selected][6]) + "% chance to\ncatch with a Pokemon"