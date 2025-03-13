#date: 2025-03-13T16:57:35Z
#url: https://api.github.com/gists/93aa26eaa0881aedcb826c17c1c8c6c4
#owner: https://api.github.com/users/RandomPigeon394

import pandas as pd
import matplotlib.pyplot as plt

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

# Filter data for low, medium, and high usage students
df_low_usage = df[df["Usage Category"] == "Low"]
df_medium_usage = df[df["Usage Category"] == "Medium"]
df_high_usage = df[df["Usage Category"] == "High"]

# Print the low usage students
if not df_low_usage.empty:
    print("\nLow usage students (less than 3 hours):")
    print(df_low_usage[["Usage Range", "Number of Students"]])
else:
    print("\nNo low usage students.")

# Print the medium usage students
if not df_medium_usage.empty:
    print("\nMedium usage students (3-7 hours):")
    print(df_medium_usage[["Usage Range", "Number of Students"]])
else:
    print("\nNo medium usage students.")

# Print the high usage students
if not df_high_usage.empty:
    print("\nHigh usage students (more than 7 hours):")
    print(df_high_usage[["Usage Range", "Number of Students"]])
else:
    print("\nNo high usage students.")
# Create a bar chart
plt.figure(figsize=(10, 6))  # Set the figure size (width, height in inches)
bars = plt.barh(df["Usage Range"], df["Number of Students"], color="skyblue")
# Add a horizontal line for the average
plt.axvline(x=df["Number of Students"].mean(), color='red', linestyle='--')

# Highlight the most common usage range
most_common_index = df["Number of Students"].idxmax()
bars[most_common_index].set_color("orange")
# Add labels and title
plt.xlabel("Number of Students")
plt.ylabel("Daily Phone Usage")
plt.title("Student Phone Usage Distribution")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Ensure the layout looks good
plt.tight_layout()

# Show the plot
plt.show()
