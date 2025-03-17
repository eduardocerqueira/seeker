#date: 2025-03-17T17:03:37Z
#url: https://api.github.com/gists/b5938599459c2002be82aaf8921b8ef6
#owner: https://api.github.com/users/suma-lang

import numpy as np
import scipy.stats as stats
import math

# Pre-test and post-test scores from the document
pretest = [62, 58, 65, 60, 66, 59, 63, 61, 64, 57, 68, 55, 67, 56, 62, 59, 65, 60, 63, 58, 66, 57, 64, 61, 62, 59, 65, 60, 63, 58]
posttest = [74, 70, 75, 71, 80, 70, 76, 70, 71, 63, 78, 63, 74, 72, 70, 71, 72, 71, 70, 61, 75, 65, 70, 70, 73, 70, 74, 71, 73, 73]

# 1. Mean calculation (Mean = ∑X/N)
def calculate_mean(data):
    return sum(data) / len(data)

pretest_mean = calculate_mean(pretest)
posttest_mean = calculate_mean(posttest)
print(f"Pre-test Mean = {pretest_mean} (Expected: 61.43333333)")
print(f"Post-test Mean = {posttest_mean} (Expected: 71.2)")

# 2. Standard Deviation calculation (SD = sqrt(∑(X-Mean)²/(n-1)))
def calculate_sd(data):
    mean = calculate_mean(data)
    variance = sum((x - mean)**2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

pretest_sd = calculate_sd(pretest)
posttest_sd = calculate_sd(posttest)
print(f"Pre-test SD = {pretest_sd} (Expected: 3.450970212)")
print(f"Post-test SD = {posttest_sd} (Expected: 4.147288271)")

# 3. Calculate differences between post-test and pre-test
differences = [post - pre for pre, post in zip(pretest, posttest)]
diff_mean = calculate_mean(differences)
diff_sd = calculate_sd(differences)
print(f"Mean of differences = {diff_mean} (Expected: 9.7667)")
print(f"SD of differences = {diff_sd} (Expected: 2.8489)")

# 4. t-test calculation: t = (x̄differences) / (Sdifferences / √n)
n = len(pretest)
se = diff_sd / math.sqrt(n)
t_stat = diff_mean / se
print(f"t-statistic = {t_stat} (Expected: 18.7773)")

# 5. Cohen's d calculation: d = |x̄d| / Sd
cohens_d = abs(diff_mean) / diff_sd
print(f"Cohen's d = {cohens_d} (Expected: 3.4283)")

# 6. Calculate p-value
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
print(f"p-value = {p_value} (Expected: 0)")

# 7. Calculate Sum of Squares (SS = ∑(X-Mean)²)
def calculate_ss(data):
    mean = calculate_mean(data)
    return sum((x - mean)**2 for x in data)

pretest_ss = calculate_ss(pretest)
posttest_ss = calculate_ss(posttest)
print(f"Pre-test Sum of Squares = {pretest_ss} (Expected: 345.36667)")
print(f"Post-test Sum of Squares = {posttest_ss} (Expected: 498.8)")

# 8. Calculate Variance (s² = SS/(n-1))
pretest_variance = pretest_ss / (n - 1)
posttest_variance = posttest_ss / (n - 1)
print(f"Pre-test Variance = {pretest_variance} (Expected: 11.909195)")
print(f"Post-test Variance = {posttest_variance} (Expected: 17.26667)")

# Output conclusion 
print("\nResults of the paired-t test indicated that there is a significant large")
print(f"difference between Before (M = {pretest_mean:.1f}, SD = {pretest_sd:.1f}) and")
print(f"After (M = {posttest_mean:.1f}, SD = {posttest_sd:.1f}), t({n-1}) = {t_stat:.1f}, p < .001.")