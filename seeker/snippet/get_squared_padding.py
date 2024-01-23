#date: 2024-01-23T17:01:36Z
#url: https://api.github.com/gists/f4a860a1291d49526700f74efc08d194
#owner: https://api.github.com/users/luca-bottero

def get_square_padding(arr):
    # Returns the array in square form with 0 padding at the end
    
    arr = arr.reshape(-1)
    l = int(np.ceil(arr.shape[0])**0.5)
    arr = np.pad(arr, (0, l*l - arr.shape[0]))
    
    return arr.reshape(l,l)