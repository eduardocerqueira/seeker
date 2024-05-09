#date: 2024-05-09T17:00:15Z
#url: https://api.github.com/gists/f938ac9c658c7f5cf4b11270dc1046e2
#owner: https://api.github.com/users/asytrao

height = float(input("Enter your height in m: "))
weight = int(input("Enter your weight in kg: "))
bmi = weight / height ** 2
if bmi < 18.5:
  print(f"Your BMI is {bmi}, you are underweight.")
elif bmi < 25:
  print(f"Your BMI is {bmi}, you have a normal weight.")
elif bmi < 30:
  print(f"Your BMI is {bmi}, you are slightly overweight.")
elif bmi < 35:
  print(f"Your BMI is {bmi}, you are obese.")
else:
  print(f"Your BMI is {bmi}, you are clinically obese.")