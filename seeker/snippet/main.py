#date: 2025-10-08T17:11:14Z
#url: https://api.github.com/gists/af5959707b974fb64cf157820c2d9223
#owner: https://api.github.com/users/muhupuameroro4-hub

# Now that I know my data types I am going to practice/ work on Typecasting, So typecasting is the process of converting a variable from one data type to another
#                                                           = Str (), int(),float(), bool()


name = "Muhupua"
age = 21
gpa = 3.8
is_student = True

#You could get the data type of the value by using the type(name) function, but you need a print statement for example
print(type(name))
print(type(age))
print(type(is_student))
print(type(gpa))

#Now I am going to convert these
#Converting my Gpa to an integer, currently it's a float. Firstly I am going to reassign the gpa and use the int() function toTypecasting to an intergar then I am going to pass tn my gpa

gpa = int(gpa)
print(gpa)
#convert my age into a floating point number
age = float(age)
print(age)
#Typecaste my age to be a string
age = str(age)
print(age)
#I will take my name variable and typecaste it into a boolean
name = bool(name)
print(name)

