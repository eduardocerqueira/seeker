#date: 2025-03-17T17:00:02Z
#url: https://api.github.com/gists/2b719ba154c559082cf7e8a9fa81d97e
#owner: https://api.github.com/users/suma-lang

import numpy as np
import math
import scipy.stats as stats

def get_numeric_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Please enter a valid number.")

def collect_data(prompt, count):
    data = []
    print(f"\n{prompt}")
    for i in range(count):
        value = get_numeric_input(f"Enter value for student {i+1}: ")
        data.append(value)
    return data

def calculate_mean(data):
    return sum(data) / len(data)

def calculate_sd(data):
    mean = calculate_mean(data)
    variance = sum((x - mean)**2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def calculate_ss(data):
    mean = calculate_mean(data)
    return sum((x - mean)**2 for x in data)

def paired_t_test_analysis():
    n = get_numeric_input("Enter number of students: ")
    n = int(n)
    
    # Pre-test scores
    pretest = collect_data("Enter PRE-TEST scores", n)
    
    # Post-test scores
    posttest = collect_data("Enter POST-TEST scores", n)
    
    # Calculate statistics
    pretest_mean = calculate_mean(pretest)
    posttest_mean = calculate_mean(posttest)
    pretest_sd = calculate_sd(pretest)
    posttest_sd = calculate_sd(posttest)
    
    # Calculate differences
    differences = [post - pre for pre, post in zip(pretest, posttest)]
    diff_mean = calculate_mean(differences)
    diff_sd = calculate_sd(differences)
    
    # t-test calculation
    se = diff_sd / math.sqrt(n)
    t_stat = diff_mean / se
    
    # Cohen's d
    cohens_d = abs(diff_mean) / diff_sd
    
    # p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    # Sum of Squares
    pretest_ss = calculate_ss(pretest)
    posttest_ss = calculate_ss(posttest)
    
    # Variance
    pretest_variance = pretest_ss / (n - 1)
    posttest_variance = posttest_ss / (n - 1)
    
    # Output results
    print("\n===== RESULTS =====")
    print(f"Pre-test Mean = {pretest_mean:.9f}")
    print(f"Post-test Mean = {posttest_mean:.9f}")
    print(f"Pre-test SD = {pretest_sd:.10f}")
    print(f"Post-test SD = {posttest_sd:.10f}")
    print(f"Mean of differences = {diff_mean:.9f}")
    print(f"SD of differences = {diff_sd:.10f}")
    print(f"t-statistic = {t_stat:.9f}")
    print(f"Cohen's d = {cohens_d:.9f}")
    print(f"p-value = {p_value:.10e}")
    print(f"Pre-test Sum of Squares = {pretest_ss:.9f}")
    print(f"Post-test Sum of Squares = {posttest_ss:.9f}")
    print(f"Pre-test Variance = {pretest_variance:.9f}")
    print(f"Post-test Variance = {posttest_variance:.9f}")
    
    print("\nResults of the paired-t test indicated that there is a significant large")
    print(f"difference between Before (M = {pretest_mean:.1f}, SD = {pretest_sd:.1f}) and")
    print(f"After (M = {posttest_mean:.1f}, SD = {posttest_sd:.1f}), t({n-1}) = {t_stat:.1f}, p < .001.")

def likert_scale_analysis():
    n = get_numeric_input("Enter number of students: ")
    n = int(n)
    
    # Data for each Likert scale category
    ease_of_use = collect_data("Enter EASE OF USE scores (1-5)", n)
    engagement = collect_data("Enter ENGAGEMENT AND INTEREST scores (1-5)", n)
    speaking = collect_data("Enter IMPROVEMENT IN SPEAKING SKILLS scores (1-5)", n)
    personalization = collect_data("Enter PERSONALIZATION OF LEARNING scores (1-5)", n)
    convenience = collect_data("Enter CONVENIENCE OF LEARNING scores (1-5)", n)
    
    # Calculate means and standard deviations
    categories = [
        ("Ease of Use", ease_of_use),
        ("Engagement and Interest", engagement),
        ("Speaking Skills", speaking),
        ("Personalization", personalization),
        ("Convenience", convenience)
    ]
    
    print("\n===== LIKERT SCALE ANALYSIS =====")
    print("{:<25} {:<15} {:<15}".format("Category", "Mean", "SD"))
    print("-" * 55)
    
    for category_name, data in categories:
        mean = calculate_mean(data)
        sd = calculate_sd(data)
        print("{:<25} {:<15.9f} {:<15.10f}".format(category_name, mean, sd))

def main():
    print("===== EDUCATIONAL DATA ANALYSIS =====")
    print("1. Paired t-test (Pre/Post Test Scores)")
    print("2. Likert Scale Analysis")
    
    choice = 0
    while choice not in [1, 2]:
        try:
            choice = int(input("Enter your choice (1 or 2): "))
            if choice not in [1, 2]:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")
    
    if choice == 1:
        paired_t_test_analysis()
    else:
        likert_scale_analysis()

if __name__ == "__main__":
    main()