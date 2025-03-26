#date: 2025-03-26T17:09:28Z
#url: https://api.github.com/gists/928868585c87d3bf72be09a9926d1aa9
#owner: https://api.github.com/users/ShayGali

import cvxpy as cp

def max_product_allocation(t):
    # Variables for the allocation
    x_A = cp.Variable()  # amount of steel Amy gets
    y_A = cp.Variable()  # amount of oil Amy gets
    
    # Values for each person
    v_A = x_A * 1 + y_A * 0  # Amy's value
    v_T = (1-x_A) * t + (1-y_A) * (1-t)  # Tammy's value
    
    # Constraints
    constraints = [
        0 <= x_A, x_A <= 1,
        0 <= y_A, y_A <= 1,
    ]

    """
    # first try - maximize product
    objective = cp.Maximize(v_A * v_T)
    problem = cp.Problem(objective, constraints)
    """

    # second try - maximize log product
    eps = 1e-6 # to avoid log(0)
    log_v_A = cp.log(v_A + eps)
    log_v_T = cp.log(v_T + eps)
    objective = cp.Maximize(log_v_A + log_v_T)
    problem = cp.Problem(objective, constraints)
    
    problem.solve()
    return x_A.value, y_A.value, v_A.value, v_T.value


def max_sum_allocation(t):
    # Variables for the allocation
    x_A = cp.Variable()  # amount of steel Amy gets
    y_A = cp.Variable()  # amount of oil Amy gets
    
    # Values for each person
    v_A = x_A * 1 + y_A * 0  # Amy's value
    v_T = (1-x_A) * t + (1-y_A) * (1-t)  # Tammy's value
    
    # Constraints
    constraints = [
        0 <= x_A, x_A <= 1,
        0 <= y_A, y_A <= 1,
    ]

    objective = cp.Maximize(v_A + v_T)
    problem = cp.Problem(objective, constraints)
    
    problem.solve()
    return x_A.value, y_A.value, v_A.value, v_T.value

def max_sqrt_sum_allocation(t):
    # Variables for the allocation
    x_A = cp.Variable()  # amount of steel Amy gets
    y_A = cp.Variable()  # amount of oil Amy gets
    
    # Values for each person
    v_A = x_A * 1 + y_A * 0  # Amy's value
    v_T = (1-x_A) * t + (1-y_A) * (1-t)  # Tammy's value
    
    # Constraints
    constraints = [
        0 <= x_A, x_A <= 1,
        0 <= y_A, y_A <= 1,
    ]

    objective = cp.Maximize(cp.sqrt(v_A) + cp.sqrt(v_T))
    problem = cp.Problem(objective, constraints)
    
    problem.solve()
    return x_A.value, y_A.value, v_A.value, v_T.value

    

if __name__ == "__main__":
    # Example of max-sum allocation (problom 4.1)
    print("Example of max-sum allocation (problom 4.1)")
    x_A, y_A, v_A, v_T = max_sum_allocation(0.3)
    print(f"Ami gets {x_A:.4f} % of steel and {y_A:.4f} % of oil.")
    print(f"Tammy gets {1-x_A:.4f} % of steel and {1-y_A:.4f} % of oil.")
    print(f"Ami's value is {v_A:.4f} and Tammy's value is {v_T:.4f}.")
    print("~"*50)

    # Example of max-sqrt-sum allocation (problom 4.2)"
    print("Example of max-sqrt-sum allocation (problom 4.2)")
    print("For t>= 0.68") # x_a = 1/t(1+t)
    x_A, y_A, v_A, v_T = max_sqrt_sum_allocation(0.8)
    print(f"Ami gets {x_A:.4f} % of steel and {y_A:.4f} % of oil.")
    print(f"Tammy gets {1-x_A:.4f} % of steel and {1-y_A:.4f} % of oil.")
    print(f"Ami's value is {v_A:.4f} and Tammy's value is {v_T:.4f}.")
    print("~"*50)
    print("For t<0.68") # x_a = 1
    x_A, y_A, v_A, v_T = max_sqrt_sum_allocation(0.2)
    print(f"Ami gets {x_A:.4f} % of steel and {y_A:.4f} % of oil.")
    print(f"Tammy gets {1-x_A:.4f} % of steel and {1-y_A:.4f} % of oil.")
    print(f"Ami's value is {v_A:.4f} and Tammy's value is {v_T:.4f}.")

    print("Example of max-product allocation (problom 4.3)")
    print("For t<=0.5") 
    x_A, y_A, v_A, v_T = max_product_allocation(0.2)
    print(f"Ami gets {x_A:.4f} % of steel and {y_A:.4f} % of oil.")
    print(f"Tammy gets {1-x_A:.4f} % of steel and {1-y_A:.4f} % of oil.")
    print(f"Ami's value is {v_A:.4f} and Tammy's value is {v_T:.4f}.")
    print("~"*50)
    print("For t>0.5")
    x_A, y_A, v_A, v_T = max_product_allocation(0.8)
    print(f"Ami gets {x_A:.4f} % of steel and {y_A:.4f} % of oil.")
    print(f"Tammy gets {1-x_A:.4f} % of steel and {1-y_A:.4f} % of oil.")
    print(f"Ami's value is {v_A:.4f} and Tammy's value is {v_T:.4f}.")
