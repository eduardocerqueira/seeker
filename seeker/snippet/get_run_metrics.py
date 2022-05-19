#date: 2022-05-19T17:35:46Z
#url: https://api.github.com/gists/875737a79b1f68edf2d566b914f29933
#owner: https://api.github.com/users/matheusfsa

import json

import numpy as np
import pandas as pd


def unroll_results(result):
    """This function unroll columns"""
    # timing
    for timing in result["timing"]:
        if timing["name"] == "execute":
            for k, v in timing.items():
                if k not in ["name", "started_at"]:
                    result[k] = v
    del result["timing"]

    # adapter response
    for k, v in result["adapter_response"].items():
        result[k] = v
    del result["adapter_response"]
    return result


if __name__ == "__main__":
    with open("target/run_results.json", encoding="utf-8") as f:
        run_results = json.load(f)
    results_json = run_results["results"]
    results = pd.DataFrame(list(map(unroll_results, results_json)))
    run_info = results.unique_id.str.split(".")
    results["run"] = run_info.str[0]

    results["target"] = np.where(
        results.run == "model", run_info.str[-1], run_info.str[-2]
    )
    results = results[results.run == "model"]
    results = results.sort_values(
        "execution_time", ascending=False
    )
    results = results.reset_index(drop=True)
    columns_to_drop = [
        "run", "code", "_message",
        "completed_at", "started_at",
        "failures", "unique_id", "message",
        "thread_id"
    ]
    results = results.drop(columns=columns_to_drop, errors='ignore')
    results.insert(0, 'target', results.pop('target'))

    tb_md = results.to_markdown()
    tb_md = "## Metrics \n" + tb_md + "\n"

    with open("target/metrics.md", "w", encoding="utf-8") as f:
        f.write(tb_md)