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
    ids=["test1", "test2"] # these are the test names for the above inputs
)
def test_my_agent(input, expected_output):
    actual_output = my_agent(input) # your code to call your agent or part of it here
    # can include static measures / evaluations here
    assert actual_output == expected_output
    # assert some other property of the output...