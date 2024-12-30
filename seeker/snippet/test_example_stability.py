#date: 2024-12-30T16:51:29Z
#url: https://api.github.com/gists/3d84b70a9df0e71048b69339d130aa62
#owner: https://api.github.com/users/skrawcz

def test_an_actions_stability():
    """Let's run it a few times to see output variability."""
    audio = ...
    outputs = [run_our_action(State({"audio": audio}))
               for _ in range(5)]
    # Check for consistency - for each key create a set of values
    variances = {}
    for key in outputs[0].keys():
        all_values = set(json.dumps(output[key]) for output in outputs)
        if len(all_values) > 1:
            variances[key] = list(all_values)
    variances_str = json.dumps(variances, indent=2)
    assert len(variances) == 0, "Outputs vary across iterations:\n" + variances_str
