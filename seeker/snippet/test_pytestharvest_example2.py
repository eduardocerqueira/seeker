#date: 2024-12-30T16:51:29Z
#url: https://api.github.com/gists/3d84b70a9df0e71048b69339d130aa62
#owner: https://api.github.com/users/skrawcz

import pytest

@pytest.mark.parametrize(
    "input, expected_output",
    [
        ("input1", "output1"),
        ("input2", "output2"),
    ],
    ids=["test1", "test2"] # can provide test names
)
def test_my_agent(input, expected_output, results_bag):
    results_bag.input = input
    results_bag.expected_output = expected_output
    results_bag.output = my_agent(input) # your code to call the agent here
    # can include static measures / evaluations here
    results_bag.success = results_bag.output == results_bag.expected_output