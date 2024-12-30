#date: 2024-12-30T16:51:29Z
#url: https://api.github.com/gists/3d84b70a9df0e71048b69339d130aa62
#owner: https://api.github.com/users/skrawcz

def test_my_agent(results_bag):
    output = my_agent("my_value")
    results_bag.input = "my_value"
    results_bag.output = output
    results_bag.expected_output = "my_expected_output"
    results_bag.exact_match = "my_expected_output" == output
    ...
    
# place this function at the end of your test module 
def test_print_results(module_results_df):
    """This function evaluates / does operations over all results captured"""
    # this will include "input", "output", "expected_output"
    print(module_results_df.columns)

    # this will show the first few rows of the results 
    print(module_results_df.head()) 

    # Add more evaluation logic here or log the results to a file, etc.
    accuracy = sum(module_results_df.exact_match) / len(module_results_df)

    # can save results somewhere
    module_results_df.to_csv(...)

    # assert some threshold of success, etc.
    assert accuracy > 0.9, "Failed overall exact match accuracy threshold"