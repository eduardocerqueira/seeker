#date: 2022-07-05T17:15:02Z
#url: https://api.github.com/gists/92a202273f13dd4de21eadf37a562990
#owner: https://api.github.com/users/akshayavb99

# Emulating switch case since it isn't already present in Python
# Source: https://youtu.be/gllUwQnYVww
# Take advantage of functions as first class objects in Python. It can be passed as objects etc.
# Overview: Create a dictionary of possible conditions, assigned to respective function that should be called

switchDict = {
    'condition1': function1,
    'condition2': function2,
    'condition3': function3
  }

def switchCase(condition, args):
  
  return switchDict.get(condition, lambda: None)(args)

def function1(args):
  return args[0] + args[1]

def function2(args):
  return args[0] - args[1]

def function3(args):
  return args[0]*args[1]

if __main__:
  answer = switchCase(function1, [1,0])
  return answer