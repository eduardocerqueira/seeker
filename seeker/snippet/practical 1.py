#date: 2023-08-10T16:58:14Z
#url: https://api.github.com/gists/6a1e54e53827736b09c18b3bd157892b
#owner: https://api.github.com/users/bishu52490

'''Q) Write a program that simulates a traffic light. The program 
should consists of following:
1. A user defined function light() that accept string as parameter 
and return 0 when parameter is RED, 1 if YELLOW , 2 if GREEN
2. A user defined function trafficLight() that accepts input from user,
display error if user input other than red yellow green; 
func light() is called and input is passed as argument, and following 
message displayed depending on return value of light():
0 --> 'STOP , your life is precious',
1 --> 'Please WAIT, till the light is Green',
2 --> 'GO! Thanks for being patient'
'SPEED THRILLS BUT KILLS' after trafficLight() is executed
'''


def trafficLight():
    msg = {0: "STOP, your life is precious",
           1: "Please WAIT, till the light turns Green",
           2:"GO! Thank you for being patient",
           -1:"Please enter a correct colour"}
    code = light(input(f"Enter a colour {colors}: ").upper())
    print(msg[code])
    print("SPEED THRILLS BUT KILLS!!")

def light(color):
    code = colors.index(color) if (color in colors) else -1
    return code
colors = ['RED',"YELLOW","GREEN"]
trafficLight()