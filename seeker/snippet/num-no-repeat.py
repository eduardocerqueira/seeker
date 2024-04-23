#date: 2024-04-23T17:09:13Z
#url: https://api.github.com/gists/42c22b8b66b67acd2b1026e225f73ed4
#owner: https://api.github.com/users/ibernabel

def count_numbers_without_adjacent_repeats():
    # Initialize count to 0
    count = 0
    
    # Loop through all possible numbers from 000000 to 999999
    for num in range(1000000):
        print(num)
        # Convert the number to a string
        num_str = str(num).zfill(6)
        
        # Check if the number has adjacent repeating digits
        has_repeats = False
        for i in range(5):
            if num_str[i] == num_str[i+1]:
                has_repeats = True
                break
        
        # If the number has no adjacent repeating digits, increment the count
        if not has_repeats:
            count += 1
        #print(num_str)
    return count
