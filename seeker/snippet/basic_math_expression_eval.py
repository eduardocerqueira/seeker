#date: 2022-01-04T17:13:33Z
#url: https://api.github.com/gists/95a234654c16abba97eeefbad90eae65
#owner: https://api.github.com/users/ryan-hallman

def evaluate(input_str):
    """
    Takes a string math equation and returns the answer without using eval
    
    @param input_str: str
        Receives a string to evaluate
    @return: int
    """
    
    operators = ['x', '/', '+', '-']
    list_of_operations = []
    temp_list = []

    for i in input_str:
        if i.isdigit():
            temp_list.append(i)
        else:
            # clear temp list and store in list of operations
            if temp_list:
                list_of_operations.append(int("".join(temp_list)))
                temp_list = []
            list_of_operations.append(i)

    if temp_list:
        list_of_operations.append(int("".join(temp_list)))

    while len(list_of_operations) > 1:
        for operator in operators:
            list_position = 0
            for val in list_of_operations:
                if val == operator:
                    list_of_operations.pop(list_position)
                    val1 = list_of_operations.pop(list_position - 1)
                    val2 = list_of_operations.pop(list_position - 1)
                    if operator == 'x':
                        new_val = val1 * val2
                    elif operator == '/':
                        new_val = val1 / val2
                    elif operator == '+':
                        new_val = val1+val2
                    else:
                        new_val = val1-val2
                    list_of_operations.insert(list_position - 1, new_val)

                list_position += 1
    return list_of_operations[0]

testcases = ["1+2", "1+2x3", "1+2x3-4", "1+2x3-4/5", "1+2x3/4+5x6"]
for case in testcases:
    print(case, "->", evaluate(case))