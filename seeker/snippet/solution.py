#date: 2024-05-07T16:53:04Z
#url: https://api.github.com/gists/be38084cebc793beff27819d61453c88
#owner: https://api.github.com/users/dominikbayerl

from pulp import *

# A list of available time slots
time_slots = ["09:00-09:30", "09:30-10:00", "10:00-10:30", "10:30-11:00", "11:00-11:30", 
"11:30-12:00", "12:00-12:30", "12:30-13:00", "13:00-13:30", "14:00-14:30"]

# A dictionary of each person's available time slots
person_time = {"person1": ["09:00-09:30", "10:00-10:30", "11:30-12:00", "12:00-12:30", "14:00-14:30"],
               "person2": ["10:00-10:30", "12:30-13:00", "13:00-13:30"],
               "person3": ["10:30-11:00", "11:00-11:30"],
               "person4": ["12:00-12:30", "09:30-10:00", "13:00-13:30", "14:00-14:30"],
               "person5": ["09:00-09:30", "11:00-11:30", "12:00-12:30"],
               "person6": ["10:00-10:30", "10:30-11:00", "13:00-13:30"]}

# Create the 'prob' variable to contain the problem data
prob = LpProblem("Appointments Scheduling Problem", LpMaximize)

# A dictionary of tuples containing all possible pairs of people and time slots
appointments = [(i, j) for i in time_slots for j in person_time.keys()] 

appointment_vars = LpVariable.dicts("appointment",(time_slots,person_time.keys()),0,1,LpBinary)

# Objective function: maximize the usage of early slots
weight = range(len(time_slots), 0, -1)
prob += lpSum([weight[i] * appointment_vars[time_slots[i]][j] for i in range(len(time_slots)) for j in person_time.keys()])

# Constraints
for person, avail_time in person_time.items(): 
    print(person, avail_time)
    prob += lpSum([appointment_vars[i][person] for i in avail_time]) == 1
    # each person only gets one appointment from their available times

# Ensure that a person can only be assigned to a timeslot if they are available
for person, avail_time in person_time.items():
    for time in time_slots:
        if time not in avail_time:
            prob += appointment_vars[time][person] == 0


for i in time_slots:
    prob += lpSum([appointment_vars[i][j] for j in person_time.keys()]) <= 1 
    # at each time slot we can have at most one appointment

# Solve the problem
prob.solve()

# Print the status of the solved LP
print("Status:", LpStatus[prob.status])

# Print the optimal scheduling of appointments
for i in prob.variables():
    if i.varValue is not None and i.varValue > 0.0:
        print(i.name, "=", i.varValue)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define a function to convert time slot string to start and end times
def convert_time_slot(time_slot):
    start_time, end_time = time_slot.split('_')
    start_hour, start_minute = map(int, start_time.split(':'))
    end_hour, end_minute = map(int, end_time.split(':'))
    return start_hour + start_minute / 60, end_hour + end_minute / 60

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(10, 2))

# Create a color map for different persons
colors = plt.cm.tab10.colors

# Loop through the variables in the problem and plot the appointments
for var in prob.variables():
    if var.varValue == 1:
        time_slot, _, person = var.name.lstrip("appointment_").rpartition("_")
        start, end = convert_time_slot(time_slot)
        rect = patches.Rectangle((start, 0.0), end-start, 1.0, 
                                 edgecolor='black', facecolor=colors[int(person[-1])-1])
        ax.add_patch(rect)
        ax.text(start + (end-start)/2, 0.4, person,
                va='center', ha='center', color='white')

# Set the x-axis labels
ax.set_xticks([i for i in range(9, 15)])
ax.set_xticklabels([f"{i}:00" for i in range(9, 15)])
ax.set_xlim(9, 15)

# Set the title and labels
ax.set_title('Appointments Timeline')
ax.set_xlabel('Time')

# Show the plot
plt.tight_layout()
plt.show()
