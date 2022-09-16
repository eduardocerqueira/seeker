#date: 2022-09-16T22:56:24Z
#url: https://api.github.com/gists/19a99b62f7a7d00f467fcf1d99413330
#owner: https://api.github.com/users/chrisgarcia001

# Construct the model
model = build_model(truck_labels,  
                    depot_labels, 
                    customer_labels, 
                    product_labels,
                    depot_product_availabilities, 
                    product_volumes, 
                    {t:trucks[t]['capacity'] for t in trucks.keys()},
                    customer_orders, 
                    {t:trucks[t]['cost_per_mile'] for t in trucks.keys()}, 
                    dist, 
                    {t:trucks[t]['base'] for t in trucks.keys()},
                    BIG_M)

# Set the initial time
start_time = time.time()

# Solve the model using the CBC solver that comes with PuLP
model.solve()

# If you have the faster CPLEX solver and want to use it, uncomment the two
# lines below and comment out the one above.
# solver = getSolver('CPLEX_CMD') 
# model.solve(solver) 

# Compute the elapsed time
elapsed_time = time.time() - start_time


# Output the total cost of transportation and computational time
print("Total Cost of Transportation = ", value(model.objective))
print("Computational Time = ", round(elapsed_time, 1), 'seconds')