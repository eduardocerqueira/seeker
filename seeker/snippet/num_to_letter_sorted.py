#date: 2023-10-25T16:45:26Z
#url: https://api.github.com/gists/7359727aebb685553c7f8639c3028b56
#owner: https://api.github.com/users/Tapanhaz

'''
The number to letter function is taken from stackoverflow .. I just make small edits 
to match the output
'''
def number_to_letter(number):
    if number >= 1 and number <= 99:
        a = [
            '', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
            'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
            'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'
        ]
        if number <= 20:
            if number % 10 == 0:
                return a[number]
            else:
                return a[number]
        elif number < 100:
            b = number - 20
            r = b % 10
            b //= 10
            return a[20 + b] + ('-' + a[r] if r != 0 else '')

def sort_key(string):
    return (len(string), string)

def prepare_list(num):
    num_list = []

    for i in range(1, num+1):
        x = number_to_letter(i)
        num_list.append(x)
    
    sorted_list = sorted(num_list, key=sort_key)
    return sorted_list

if __name__ == "__main__":
    my_list = prepare_list(99)
    print(my_list)
