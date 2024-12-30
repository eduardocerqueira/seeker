#date: 2024-12-30T16:51:29Z
#url: https://api.github.com/gists/3d84b70a9df0e71048b69339d130aa62
#owner: https://api.github.com/users/skrawcz

import pytest
from our_agent_application import prompt_for_more

from burr.core import state

# the following is required to run file based tests
from burr.testing import pytest_generate_tests  # noqa: F401

@pytest.mark.file_name("prompt_for_more.json") # our fixture file with the expected inputs and outputs
def test_an_agent_action(input_state, expected_state, results_bag):
    """Function for testing an individual action of our agent."""
    input_state = state.State.deserialize(input_state)
    expected_state = state.State.deserialize(expected_state)
    _, output_state = prompt_for_more(input_state)  # exercising an action of our agent

    results_bag.input_state = input_state
    results_bag.expected_state = expected_state
    results_bag.output_state = output_state
    results_bag.foo = "bar"
 
    # TODO: choose appropriate way to evaluate the output
    # e.g. exact match, fuzzy match, LLM grade, etc.
    # this is exact match here on all values in state
    exact_match = output_state == expected_state

    # for output that varies, you can do something like this
    # assert 'some value' in output_state["response"]["content"]
    # or, have an LLM Grade things -- you need to create the llm_evaluator function:
    # assert llm_evaluator("are these two equivalent responses. Respond with Y for yes, N for no",
    # output_state["response"]["content"], expected_state["response"]["content"]) == "Y"
    # store it in the results bag
    results_bag.correct = exact_match

    # place any asserts at the end of the test
    assert exact_match