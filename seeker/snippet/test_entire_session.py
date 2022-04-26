#date: 2022-04-26T17:16:09Z
#url: https://api.github.com/gists/3ad055bd20929dd8c2378c15e873e7c1
#owner: https://api.github.com/users/halflearned

import pickle
from datetime import datetime, timezone
from itertools import chain

import numpy as np
import pytest

from sleep_x_ml_reconciliation_service import SleepXMLReconciliation
from sleep_x_ml_reconciliation_service.constants import (
    AWAKE_STAGE,
    DEEP_STAGE,
    LARGE_SLEEP_GAP_FLAG,
    LIGHT_STAGE,
    NO_INTENT_TO_SLEEP_FLAG,
    PUT_GAP_TOLERANCE_SECONDS,
    REM_STAGE,
    SLEEP_GAP_TOLERANCE_SECONDS,
    UNINTENDED_USER_FLAG,
)
from sleep_x_ml_reconciliation_service.exceptions import (
    IncorrectArchitectureException,
    IncorrectPredictionTimeException,
)
from sleep_x_ml_reconciliation_service.utils.formatting import format_output, to_timestamp
from sleep_x_ml_reconciliation_service.utils.validation import (
    validate_put_response,
    validate_sleep_put_overlap,
)


def find_chunk(sequence, point):
    """
    Auxiliary function that retrieves the chunk in sequence satistifying
    chunk['start'] <= point <= chunk['end']
    """
    found = [chunk for chunk in sequence if chunk["start"] <= point <= chunk["end"]]
    if found:
        # Sometimes there are multiple chunks that overlap.
        # In this case, take one with the latest start.
        return max(found, key=lambda x: x["start"])
    else:
        return []


def from_timestamp(stamp):
    """Reverts timestamp back into integer, i.e. '2001-09-09T01:46:40.000Z' into 1000000000000"""
    return int(
        1000
        * datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()
    )


def format_halo_labels(sleep_predictions):
    output = []
    for chunk in sleep_predictions:
        label = chunk["stage"].lower()
        if label in ["wake", "awake"]:
            stage = AWAKE_STAGE
        elif label in ["rem"]:
            stage = REM_STAGE
        elif label in ["deep"]:
            stage = DEEP_STAGE
        elif label in ["light"]:
            stage = LIGHT_STAGE
        else:
            raise ValueError(f"Unknown label {chunk['stage']}")

        output.append({"start": chunk["start"], "end": chunk["end"], "stage": stage})
    return output


def group(predictions, key):
    predictions = sorted(predictions, key=lambda x: x["start"])
    output = [predictions[0]]
    for chunk in predictions[1:]:
        if chunk["start"] <= output[-1]["end"] and chunk[key] == output[-1][key]:
            output[-1]["end"] = chunk["end"]
        else:
            output.append(chunk)
    return output


def get_before_and_after(sleep_predictions, point):
    before = [chunk for chunk in sleep_predictions if chunk["end"] <= point]
    after = [chunk for chunk in sleep_predictions if chunk["start"] >= point]
    return before, after


def test_holistic(mocker):
    """
    Heuristic real data from sessions stored in the alpha account.
    This simple test does not test for correctness, only checks the the code runs.
    """
    # Instantiating with whatever default values of currently have.
    sleepxml = SleepXMLReconciliation()

    with open("/Users/hadadv/tmp/device-alpha", "rb") as f:
        example_data = pickle.load(f)

    for i, session_id in enumerate(example_data):
        if i % 50 == 0:
            print(i)

        data = example_data[session_id]

        sleep_predictions = data.get("sleep", False)
        put_predictions = data.get("put", False)

        if not sleep_predictions or not put_predictions:
            continue

        put_response = [{"state_details_list": group(put_predictions, "state")}]
        sleep_predictions = group(format_halo_labels(sleep_predictions), "stage")

        sleep_start = sleep_predictions[0]["start"]
        sleep_end = sleep_predictions[0]["end"]
        its_predictions = [
            {
                "start": sleep_start,
                "end": sleep_end,
                "its_probs": [0.5] * int(np.ceil((sleep_end - sleep_start) / 30_000)),
            }
        ]

        mocker.patch(
            "sleep_x_ml_reconciliation_service.reconciliation.PrimaryUserTrackingDDB.get_data",
            return_value=put_response,
        )
        mocker.patch(
            "sleep_x_ml_reconciliation_service.reconciliation.SleepStageDDB.get_data",
            return_value=sleep_predictions,
        )
        mocker.patch(
            "sleep_x_ml_reconciliation_service.reconciliation.IntentToSleepDDB.get_data",
            return_value=its_predictions,
        )

        try:
            reconciled = sleepxml.get_reconciled_data(session_id)

        # If any exception needs to be raised,
        # ensure they are raised appropriately.
        except IncorrectArchitectureException:
            with pytest.raises(IncorrectArchitectureException):
                validate_put_response(put_response)
            continue

        except IncorrectPredictionTimeException:
            with pytest.raises(IncorrectPredictionTimeExpression):
                validate_sleep_put_overlap(sleep_predictions, put_predictions)
            continue

        # If everything was thrown out in reconciliation, just move on.
        if not reconciled:
            continue

        # Any other exception would raise an error here.
        # The rest of the code heuristically checks for correctness.

        # If the entire session is marked as UU, check that reconciled is empty.
        if all(chunk["state"] == "UU" for chunk in put_predictions):
            assert sleep_predictions == []
            continue

        sleep_predictions = sorted(sleep_predictions, key=lambda x: x["start"])
        put_predictions = sorted(put_predictions, key=lambda x: x["start"])

        # Otherwise, get start and end of reconciled session.
        recon_start = from_timestamp(reconciled[0]["startTime"])
        recon_end = from_timestamp(reconciled[-1]["endTime"])

        # Check that reconciled data only exists for the intervals where
        # sleep and PUT predictions are both available.
        start_available = max(sleep_predictions[0]["start"], put_predictions[0]["start"])
        end_available = min(sleep_predictions[-1]["end"], put_predictions[-1]["end"])
        assert recon_start >= start_available
        assert recon_end <= end_available

        # Check that reconciled data does not end on an interval marked as UU.
        # Disabling as this test seems to raise false negatives. Needs investigation.
        # put_at_recon_end = find_chunk(put_predictions, recon_end - 100)["state"]
        # assert put_at_recon_end == "IU"

        # Check correctness of reconciled sleep predictions.
        for recon in reconciled:

            # Pick a point in that reconciled chunk.
            recon_midpoint = (
                from_timestamp(recon["endTime"]) + from_timestamp(recon["startTime"])
            ) // 2

            # Find the corresponding PUT chunk at that time.
            put = find_chunk(put_predictions, recon_midpoint)
            # Find the corresponding sleep chunk that time.
            sleep = find_chunk(sleep_predictions, recon_midpoint)

            # If PUT state was UU, recon should be either UNINTENDED_USER_FLAG or NO_INTENT_TO_SLEEP_FLAG
            if put and sleep and put["state"] == "UU":
                assert recon["stage"] in [UNINTENDED_USER_FLAG, NO_INTENT_TO_SLEEP_FLAG]

            # If PUT state is IU and there was sleep data, recon should be equal to sleep or NO_INTENT_TO_SLEEP_FLAG
            if put and sleep and put["state"] == "IU":
                assert recon["stage"] in [sleep["stage"], NO_INTENT_TO_SLEEP_FLAG]

            # If PUT state is IU and there was no sleep data, recon depends on gap.
            if put and not sleep:

                # If gap is large, recon should be marked as such.
                if recon["stage"] == LARGE_SLEEP_GAP_FLAG:
                    # Assert there was really such a gap
                    before, after = get_before_and_after(sleep_predictions, recon_midpoint)
                    gap_seconds = (after[0]["start"] - before[-1]["end"]) / 1_000
                    assert gap_seconds > SLEEP_GAP_TOLERANCE_SECONDS

                # If it is marked as AWAKE, then put must be UU
                elif recon["stage"] == UNINTENDED_USER_FLAG:
                    assert put["state"] == "UU"

                # If gap is small, recon stage should be equal to value of next chunk.
                else:
                    # Find the value of the next chunk.
                    next_sleep = [
                        chunk for chunk in sleep_predictions if chunk["start"] >= recon_midpoint
                    ]
                    assert recon["stage"] == next_sleep[0]["stage"]
