#date: 2024-12-02T17:06:52Z
#url: https://api.github.com/gists/1148009e518ca01a63e4bf13d2f02769
#owner: https://api.github.com/users/Fuanyi-237

#Code to create a Diaamond with a letter starting from "A"(to that letter and back)
def rows(letter):
    #collecting the number of rows from the alpha-numeric correspondant
    row = int(ord(letter)) - 64
    result = []
    n = 0
    spaces = row - 2   #front spaces
    space_btw = 2*n + 1 #spaces in between the letter
    if row > 1: #I wish to do a logic from the second row to the last
        # firstly, I secure the first letter in the list(result) 
        first_letter = " " * (row-1) + "A" + " " * (row-1) 
        result.append(first_letter)
        #Collect from 'B' to last letter
        for i in range(ord('B'), ord(letter) + 1): 
            #logic behind collecting the letters in a row
            row_letter = " "*(spaces) + chr(i) + " "*(space_btw) + chr(i) + " "*(spaces)  
            result.append(row_letter)
            space_btw += 2
            spaces -= 1
        space_btw -= 2 
        spaces += 1
        #Collect from second to last letter to B
        for k in range(ord(letter)-1, ord('A'), -1):
            spaces += 1
            space_btw -= 2
            row_letter = " "*(spaces) + chr(k) + " "*(space_btw) + chr(k) + " "*(spaces)
            result.append(row_letter)
        #secure my last letter, which is same as first
        last_letter = " " * (row-1) + "A" + " " * (row-1)
        result.append(last_letter)
        return result          
    else:
        return [chr(ord(letter))]