#date: 2023-07-20T16:59:58Z
#url: https://api.github.com/gists/0861a4f793676fcc80f4df9ac816a08f
#owner: https://api.github.com/users/ShahaneMirzoian

conversions = [39, 27.4, 20.6, 11, 2]
steps = ["Website visit", "Downloads", "Potential customers", 
         "Requested price", "invoice sent"]
colors = ('#FFCFD2', '#FFCFD2', '#FFCFD2', '#FFCFD2')

fig = go.Figure(go.Funnel(x=conversions,
                          y=steps))
fig.update_layout(height=390, width=550,  
                  colorway = colors, 
                  template='simple_white')

fig.update_yaxes(showline=False)
fig.show()