#date: 2023-07-03T16:57:24Z
#url: https://api.github.com/gists/8286caff5d6d9be036bf4469c9f74662
#owner: https://api.github.com/users/DiogoRibeiro7

from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus

# Dynamic data: task names and their respective time requirements
tasks = ['Task1', 'Task2', 'Task3', 'Task4', 'Task5']
time_requirements = {
    'Task1': {'Laser': 2, 'Iron Mark': 3, 'Ink': 4},
    'Task2': {'Laser': 3, 'Iron Mark': 2, 'Ink': 5},
    'Task3': {'Laser': 4, 'Iron Mark': 1, 'Ink': 6},
    'Task4': {'Laser': 1, 'Iron Mark': 5, 'Ink': 3},
    'Task5': {'Laser': 2, 'Iron Mark': 4, 'Ink': 2},
}

# Dynamic data: number of machines available for each type
machines_available = {
    'Laser': 1,
    'Iron Mark': 5,
    'Ink': 5,
}

# Dynamic data: stock availability for each material
stock_availability = {
    'Material1': 10,
    'Material2': 8,
    'Material3': 15,
}

# Create the LP problem
problem = LpProblem("MachineTimeAllocation", LpMaximize)

# Define decision variables
allocation = LpVariable.dicts('allocation', [(task, machine) for task in tasks for machine in machines_available], 0, 1, LpInteger)

# Define the objective function
problem += lpSum([allocation[task, machine] for task in tasks for machine in machines_available]), "Total Production"

# Add the constraints
for task in tasks:
    problem += lpSum([allocation[task, machine] for machine in machines_available]) == 1, f"AllocationConstraint_{task}"

for machine, available_count in machines_available.items():
    problem += lpSum([allocation[task, machine] for task in tasks]) <= available_count, f"MachineAvailability_{machine}"

for material, required_stock in stock_availability.items():
    problem += lpSum([allocation[task, machine] * time_requirements[task][machine] for task in tasks for machine in machines_available if material in time_requirements[task]]) <= required_stock, f"StockAvailability_{material}"

# Solve the problem
problem.solve()

# Print the status of the solution
print("Status:", LpStatus[problem.status])

# Print the optimal allocation of machine time
for task in tasks:
    for machine in machines_available:
        if allocation[task, machine].varValue == 1:
            print(f"Task '{task}' allocated to {machine}")

# Print the total production achieved
print("Total Production:", problem.objective.value())
