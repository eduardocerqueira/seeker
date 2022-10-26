#date: 2022-10-26T16:56:14Z
#url: https://api.github.com/gists/168f95a2c80d1cdcb725ae30178fbf54
#owner: https://api.github.com/users/criticalth

def export_list(list_exp, txt_file):

    with open(txt_file, 'w', encoding="utf-8") as filehandle:
        for listitem in list_exp:
            filehandle.write(f'{listitem}\n')


def import_list(txt_file):
    # Define an empty list
    res = []

    # Open the file and read the content in a list
    with open(txt_file, 'r', encoding="utf-8") as filehandle:
        for line in filehandle:
            # Remove linebreak which is the last character of the string
            curr_place = line[:-1]
            # Add item to the list
            res.append(curr_place)

    return res
  