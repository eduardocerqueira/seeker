#date: 2023-04-06T16:41:32Z
#url: https://api.github.com/gists/a830d7fe21bfe19dc31ea37df8bcfead
#owner: https://api.github.com/users/larsoner

# Check 1-tailed paired t bias when selecting ROIs using different methods
import numpy as np
from scipy.stats import ttest_1samp

rng = np.random.default_rng(0)

n_subjects = 20
n_sensors = 100
n_run = 5000  # number of times to do this
prop_false = np.zeros((n_run, 3, 2))
n_roi = n_sensors // 20
for ri in range(n_run):
    A = rng.normal(size=(n_subjects, n_sensors))
    B = rng.normal(size=(n_subjects, n_sensors))
    sum_ = A + B
    diff = A - B
    for ai, average_sensors in enumerate((False, True)):
        # naive: just compute all stats
        check = diff
        if average_sensors:
            check = check.mean(-1)
        _, p = ttest_1samp(diff, 0, alternative='greater')
        prop_false[ri, 0, ai] = (p < 0.05).sum() / p.size
        # take top 5% of mean activation sum then compute stats on difference
        roi = np.argsort(sum_.mean(axis=0))[::-1][:n_roi]
        check = diff[:, roi]
        if average_sensors:
            check = check.mean(-1)
        _, p = ttest_1samp(check, 0, axis=0, alternative='greater')
        prop_false[ri, 1, ai] = (p < 0.05).sum() / p.size
        # take top 5% of mean activate difference then compute stats on difference
        roi = np.argsort(diff.mean(axis=0))[::-1][:n_roi]
        check = diff[:, roi]
        if average_sensors:
            check = check.mean(-1)
        _, p = ttest_1samp(check, 0, axis=0, alternative='greater')
        prop_false[ri, 2, ai] = (p < 0.05).sum() / p.size

with np.printoptions(precision=2, suppress=True):
    print('False alarm percent (should be 5%)')
    print('Left: sensors independently tested; right: averaged across them')
    for ki, kind in enumerate(('naive', 'sum', 'diff')):
        print(f'- {kind}: {100 * prop_false.mean(axis=0)[ki]}')
