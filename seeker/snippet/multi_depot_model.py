#date: 2022-09-16T22:46:09Z
#url: https://api.github.com/gists/9022d188130674a7da579733c57478c7
#owner: https://api.github.com/users/chrisgarcia001

# The inputs for this function are as follows:
# trucks: A list of distinct truck labels (e.g., ['T1', 'T2', ...])
# depots: A list of distinct depot labels (e.g., ['D1', 'D2', ...])
# customers: A list of distinct customer labels (e.g., ['C1', 'C2', ...])
# products: A list of distinct product labels (e.g., ['C1', 'C2', ...])
# prod_availability: A dict of form {<depot label>:{<product label>:<units available>, ...}, ...}
# prod_volume: A dict of form {<product label>:<volume>}
# truck_cap: A dict of form {<truck label>:<volume capacity>}
# demand: A dict of form {<customer label>}:{<product label>:<units ordered>, ...}, ...}
# cost_per_mile: A dict of form {<truck label>:<cost per mile>}
# dist_matrix: dict of form {(<location label>, <location label>):<distance>, ...}. Locations are depots + customers
# truck_base: A dict of form {<truck label>:<depot label>}
# big_m: An arbitrarily large constant.
def build_model(trucks, depots, customers, products, prod_availability,
                prod_volume, truck_cap, demand, cost_per_mile,
                dist_matrix, truck_base, big_m):
    H = trucks
    I = depots
    J = customers
    K = products
    L = I + J # Set of all distinct locations
    a = prod_availability
    e = prod_volume
    E = truck_cap
    r = demand
    d = dist_matrix
    c = cost_per_mile
    b = {h:{i:1 if truck_base[h] == i else 0 for i in I} for h in H}
    
    # u[h][j][k] = Units of product k delivered to customer j on truck h
    u = LpVariable.dicts("u", (H, J, K), 0, None, LpInteger) 
    
    # x[h][i][j] == 1 if truck h travels directly from location i to j, 0 otherwise
    x = LpVariable.dicts("x", (H, L, L), 0, 1, LpInteger)
    
    prob = LpProblem("MultiDepot_VRP", LpMinimize)
    prob += (lpSum([c[h] * d[i,j] * x[h][i][j] for h in H for i in L for j in L]), 
                   'Total_Cost') # Objective Function
    for h in H:
        prob += (lpSum([e[k] * u[h][j][k] for j in J for k in K]) <= E[h]) # Ensure no truck capacity exceeded
        for i in L:
            prob += (lpSum([x[h][j][i] for j in L]) == lpSum([x[h][i][j] for j in L])) # For each loc, truck in -> truck out
        for i in I:
            prob += (lpSum([x[h][i][j] for j in L]) <= b[h][i]) # Ensure no truck leaves a non-base depot
        for W in allcombinations(J, len(J)): 
            prob += (lpSum([x[h][i][j] for i in W for j in W]) <= len(W) - 1) # Ensure no subset of customers contains a circuit.
                      
    for k in K:
        for i in I:
            prob += (lpSum([b[h][i] * u[h][j][k] for h in H for j in J]) <= a[i][k]) # No depot ships more of a product than it has
        for j in J:
            prob += (lpSum(u[h][j][k] for h in H) == r[j][k]) # Each customer gets the products they ordered
            for h in H:
                prob += (u[h][j][k] <= big_m * lpSum(x[h][i][j] for i in L)) # No truck carries products for a customer unless it visits the customer.
    return prob
    