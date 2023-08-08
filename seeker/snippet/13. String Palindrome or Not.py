#date: 2023-08-08T16:59:28Z
#url: https://api.github.com/gists/8154c2bdbf978027a56faeec9e729df4
#owner: https://api.github.com/users/Vigneswarsiddu

# Method 1: Using String Slicing

String=(input("Enter the name: "))
reverse=(String[::-1])
if(String==reverse):
    print("palindrome")
else:
    print("Not palindrome")