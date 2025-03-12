#date: 2025-03-12T16:54:32Z
#url: https://api.github.com/gists/086cd3d78e31b8d3b139c4d8307cb6f7
#owner: https://api.github.com/users/RandomPigeon394

import pandas as pd

# Data of phone usage
phone_data = {
    "Usage Range": ["0-1 hours", "1-3 hours", "3-5 hours", "5-7 hours", "7-9 hours", "9-11 hours", "11+ hours"],
    "Number of Students": [0, 2, 3, 2, 3, 0, 1],
    "Usage Range Numbers": [0.5, 2, 4, 6, 8, 10, 11] 
}

df = pd.DataFrame(phone_data)

# Add Midpoint Column
df["Midpoint"] = [0.5, 2, 4, 6, 8, 10, 11]  # Midpoint of each range

# Calculate total hours spent on phone by all students
df["Total Hours"] = df["Midpoint"] * df["Number of Students"]
total_hours_spent = df["Total Hours"].sum()

# Usage levels
def categorize_usage(midpoint):
    if midpoint < 3:
        return "Low"
    elif midpoint < 7:
        return "Medium"
    else:
        return "High"

df["Usage Category"] = df["Midpoint"].apply(categorize_usage)

# Filters data where phone usage is greater than 5 hours
df_filtered = df[df["Usage Range Numbers"] > 5]

# Calculates the percentages of students using phones more than 5 hours
total_students = df["Number of Students"].sum()
students_above_5_hours = df_filtered["Number of Students"].sum()

if total_students > 0:  
    percentage = (students_above_5_hours / total_students) * 100
    print(f"\nPercentage of students using phones more than 5 hours: {percentage:.2f}%")
else:
    print("\nNo students in the dataset.")

# Finds the median phone usage hours
usage_expanded = []
for i in range(len(df)):
    usage_expanded.extend([df["Usage Range Numbers"][i]] * df["Number of Students"][i])

if usage_expanded:  
    median_usage = pd.Series(usage_expanded).median()
    print(f"\nMedian phone usage: {median_usage:.2f} hours")
else:
    print("\nNo student data available to calculate the median.")

# Identify the most common phone usage range
row = df["Number of Students"].idxmax()
print(f"\nMost common phone usage: {df['Usage Range'][row]} "
      f"with {df['Number of Students'][row]} students.")

# Display total hours spent
print(f"\nTotal hours spent on phones by all students: {total_hours_spent:.2f} hours")
