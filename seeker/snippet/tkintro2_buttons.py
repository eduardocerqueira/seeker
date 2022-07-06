#date: 2022-07-06T16:55:29Z
#url: https://api.github.com/gists/615a0139be0b55abb52fb2112c1e31da
#owner: https://api.github.com/users/Pinnacle55

### Search Frame
search_frame = tk.Frame(window, relief = tk.RAISED, borderwidth = 2)
search_frame.columnconfigure([0,1,2,3], weight = 1)

# Convert this to a button
left_button = tk.Button(search_frame, text = "Left Arrow", font = ("Futura", 16))
left_button.grid(row = 0, column = 0)

# Expand the label to an Entry-Button combo
# Give it a relevant name so that we can use it.
submit_entry = tk.Entry(search_frame, font = ("Futura", 16))
submit_entry.grid(row = 0, column = 1)

submit_button = tk.Button(search_frame, text = "Search!", font = ("Futura", 16))
submit_button.grid(row = 0, column = 2)

# Convert this to a button
right_button = tk.Button(search_frame, text = "Right Arrow", font = ("Futura", 16))
right_button.grid(row = 0, column = 3)

search_frame.grid(row = 0, column = 1, columnspan = 2, sticky = "ew")