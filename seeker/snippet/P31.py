#date: 2024-06-05T17:01:46Z
#url: https://api.github.com/gists/73cd66198c51e74aa8533d760d159a57
#owner: https://api.github.com/users/VedaCPatel

#printing patterns using loops-3
n=int(input("Enter a number: "))
for i in range(1,n+1):
    print("* "*i)
for j in range(0,n):
    count = n - 1 - j
    print("* "*count)
    count-=1


