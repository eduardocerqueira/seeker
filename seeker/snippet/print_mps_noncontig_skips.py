#date: 2026-02-06T17:27:31Z
#url: https://api.github.com/gists/b2c2b291397a1f9f8e7347230e1a45d8
#owner: https://api.github.com/users/malfet

#!/usr/bin/env python3
"""Print all ops with MPS skips for non-contiguous input."""

import unittest
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.opinfo.core import DecorateInfo


def main():
    ops_with_mps_noncontig_skip = []

    for op in op_db:
        for skip in op.skips:
            # Skip entries that aren't DecorateInfo
            if not isinstance(skip, DecorateInfo):
                continue

            # Check if this skip is for test_noncontiguous_samples on MPS
            if (
                skip.test_name == "test_noncontiguous_samples"
                and skip.device_type == "mps"
            ):
                # Determine skip type from decorators
                skip_type = "skip"
                for dec in skip.decorators:
                    if dec == unittest.expectedFailure:
                        skip_type = "xfail"
                    elif hasattr(dec, "__name__") and "skip" in dec.__name__.lower():
                        skip_type = "skip"

                op_name = op.name
                if op.variant_test_name:
                    op_name = f"{op.name}.{op.variant_test_name}"

                ops_with_mps_noncontig_skip.append((op_name, skip_type, skip.dtypes))

    # Print results
    print(f"Found {len(ops_with_mps_noncontig_skip)} ops with MPS non-contiguous skips:\n")
    for op_name, skip_type, dtypes in sorted(ops_with_mps_noncontig_skip, key=lambda x: x[0]):
        dtype_str = f" (dtypes: {dtypes})" if dtypes else ""
        print(f"  {op_name}: {skip_type}{dtype_str}")


if __name__ == "__main__":
    main()
