#date: 2021-10-29T17:13:46Z
#url: https://api.github.com/gists/9bec83872329d3f58b6f32172c997762
#owner: https://api.github.com/users/TuringNPcomplete

#Cropping different parts of the reptile image

#Making row slices
show_images([reptile_arr[int(row_sz*start_mult): int(row_sz*(start_mult+0.2)), :] for start_mult in [0, 0.2, 0.4, 0.6, 0.8]])

#Making column slices
show_images([reptile_arr[:, int(row_sz*start_mult): int(row_sz*(start_mult+0.2))] for start_mult in [0, 0.2, 0.4, 0.6, 0.8]])