#date: 2024-06-17T16:59:32Z
#url: https://api.github.com/gists/c836162a8de902a560a5103dd4566787
#owner: https://api.github.com/users/chrishavlin

"""
to store:

    $ python manual_grid_val_test.py 1

to compare

    $ python manual_grid_val_test.py 0

"""
import yt
import json
import sys
from yt.utilities.answer_testing.framework import GridValuesTest


if __name__ == "__main__":

    store_answers = int(sys.argv[1]) == 1
    fld = ('gas', 'velocity_divergence')
    ds_fn = "IsolatedGalaxy/galaxy0030/galaxy0030"
    hash_file = 'iso_gal_grid_val_hashes.json'

    gvt = GridValuesTest(ds_fn, fld)

    new_hashes = gvt.run()

    if store_answers:
        print("writing answers")
        with open(hash_file, 'w') as fp:
            json.dump(new_hashes, fp)
    else:
        print("loading old answers")
        with open(hash_file, 'r') as fp:
            old_hashes = json.load(fp)
        old_hashes_valid = {}
        for k, val in old_hashes.items():
            old_hashes_valid[int(k)] = val
        print("comparing")
        gvt.compare(new_hashes, old_hashes_valid)
        print("success")
