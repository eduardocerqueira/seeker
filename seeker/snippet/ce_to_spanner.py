#date: 2023-04-19T16:54:02Z
#url: https://api.github.com/gists/23cd94915413bc0fd510df443842d9d7
#owner: https://api.github.com/users/Intelrunner

# use at own risk, not verified by engineer

# Prompt user for inputs
num_cores = int(input("Enter the number of cores per machine: "))
ram_gb = int(input("Enter the amount of RAM (in GB) per machine: "))
disk_size_gb = int(input("Enter the size of disk (in GB) per machine: "))
disk_type = input("Enter the type of disk (ssd/standard/balanced): ")
num_machines = int(input("Enter the number of machines: "))
num_spanner_nodes = int(input("Enter the number of estimated Spanner nodes: "))
region = input("Enter the region: ")

# Create object from input data
instance_data = {
    "num_cores": num_cores,
    "ram_gb": ram_gb,
    "disk_size_gb": disk_size_gb,
    "disk_type": disk_type,
    "num_machines": num_machines,
    "region": region,
}
spanner_data = {
    "num_spanner_nodes": num_spanner_nodes,
    "region": region,
}

# Define formula for calculating instance cost
def calculate_instance_cost(num_cores, ram_gb, disk_size_gb, disk_type):
    cost_per_hour = 0.0
    if disk_type == "ssd":
        cost_per_hour += 0.17
    elif disk_type == "standard":
        cost_per_hour += 0.1
    elif disk_type == "balanced":
        cost_per_hour += 0.13
    else:
        print("Invalid disk type")
        return None
    
    cost_per_hour += (num_cores * 0.046) + (ram_gb * 0.004)
    cost_per_day = cost_per_hour * 24
    return cost_per_day

# Define formula for calculating Spanner storage cost
def calculate_spanner_storage_cost(num_spanner_nodes):
    cost_per_month = num_spanner_nodes * 0.18
    return cost_per_month

# Define formula for calculating Spanner nodes cost
def calculate_spanner_nodes_cost(num_spanner_nodes):
    cost_per_hour = num_spanner_nodes * 0.7
    cost_per_day = cost_per_hour * 24
    return cost_per_day

# Calculate costs
instance_cost = calculate_instance_cost(num_cores, ram_gb, disk_size_gb, disk_type) * num_machines
spanner_storage_cost = calculate_spanner_storage_cost(num_spanner_nodes)
spanner_nodes_cost = calculate_spanner_nodes_cost(num_spanner_nodes)

# Print results
print("Instance cost per day: ${:.2f}".format(instance_cost))
print("Spanner storage cost per month: ${:.2f}".format(spanner_storage_cost))
print("Spanner nodes cost per day: ${:.2f}".format(spanner_nodes_cost))
