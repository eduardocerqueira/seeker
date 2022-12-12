#date: 2022-12-12T17:03:14Z
#url: https://api.github.com/gists/816b0115149afa199b46a0ba8d005559
#owner: https://api.github.com/users/abhishekchandra2522k

from datetime import datetime

user_in = input("enter your goal with a deadline separated by colon\n")
input_list = user_in.split(":")

goal = input_list[0]
deadline = input_list[1]

deadline_date = datetime.strptime(deadline, "%d.%m.%Y")
today_date = datetime.today()

# calculate how many days from now till deadline
time_till = deadline_date - today_date

print(f"Dear user! Time remaining for your goal ({goal}): {time_till.days} days")
hours_till = int(time_till.total_seconds() / 60 / 60)
print(f"Dear user! Time remaining for your goal ({goal}): {hours_till} hours")