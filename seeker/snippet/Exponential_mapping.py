#date: 2024-05-21T16:56:59Z
#url: https://api.github.com/gists/544b645b0f88116662472fb828ef8183
#owner: https://api.github.com/users/ameerfayiz

import math

def exponential_mapping(x,decay_factor=5,input_initial=1,input_final=60,output_initial=150,output_final=5):

    # Normalizing x from range 1 to 60 to 0 to 1
    normalized_x = (x - input_initial) / (input_final - input_initial)
    
    # Exponential decay function
    y = math.exp(-normalized_x * decay_factor)  # Choose a decay rate that fits well, here using 5
    
    # Scaling y to the desired range 120 to 5
    output = output_final + (output_initial - output_final) * y
    
    return output

# Test the function
for i in range(1, 61):
    print(f"Input: {i}, Output: {exponential_mapping(i):.2f}")