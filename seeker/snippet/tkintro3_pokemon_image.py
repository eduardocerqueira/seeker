#date: 2022-07-30T19:05:56Z
#url: https://api.github.com/gists/98114aa5b11f6f62781bf8ae0078a360
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
    
    # Display the Pokemon's image
    label_image = ImageTk.PhotoImage(file = \
                 f"Pokemon Pictures/{pokedex[pokedex.name == selected].index[0] + 1}.png")
    pic_label.config(image = label_image)
    # Remember to keep a reference to the image or it won't display!
    pic_label.image = label_image