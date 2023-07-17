#date: 2023-07-17T16:58:55Z
#url: https://api.github.com/gists/1a2ff46122bd398e31571096e54f43fc
#owner: https://api.github.com/users/AsifAlFaisal


def palindrome(x: int) -> bool:

    if -231 <= x <= 231-1:    
        if x<0:
            return False
        
        if str(x) ==str(x)[::-1]:
            return True
        else:
            return False
    else:
        return False


if __name__=="__main__":
    x = int(input("Enter Your Number: "))   
    print(palindrome(x))