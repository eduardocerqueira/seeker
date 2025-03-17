#date: 2025-03-17T17:10:17Z
#url: https://api.github.com/gists/952319f5b39224bbd8d24d423bb8bca2
#owner: https://api.github.com/users/suma-lang

import numpy as np
import math

# Likert scale responses for all 30 students
ease_of_use = [4.2, 3.8, 4.5, 4.1, 4.6, 3.9, 4.3, 4.2, 4.4, 4.3, 4.3, 3.8, 4.5, 4.1, 4.6, 3.9, 4.3, 4.2, 4.4, 4.3, 4.2, 3.8, 4.5, 4.1, 4.6, 3.9, 4.3, 4.2, 4.4, 4.3]

engagement_interest = [4.5, 4.3, 4.7, 4.5, 4.8, 4.2, 4.6, 4.3, 4.7, 4.4, 4.4, 4.3, 4.7, 4.5, 4.8, 4.2, 4.6, 4.3, 4.7, 4.4, 4.5, 4.3, 4.7, 4.5, 4.8, 4.2, 4.6, 4.3, 4.7, 4.4]

speaking_skills = [4.1, 3.7, 4.4, 4.1, 4.3, 3.8, 4.1, 3.9, 4.2, 4.0, 4.2, 3.7, 4.4, 4.1, 4.3, 3.8, 4.1, 3.9, 4.2, 4.0, 4.1, 3.7, 4.4, 4.1, 4.3, 3.8, 4.1, 3.9, 4.2, 4.0]

personalization = [4.0, 3.9, 4.3, 4.0, 4.4, 3.9, 4.2, 4.0, 4.3, 4.1, 4.2, 3.9, 4.3, 4.0, 4.4, 3.9, 4.2, 4.0, 4.3, 4.1, 4.0, 3.9, 4.3, 4.0, 4.4, 3.9, 4.2, 4.0, 4.3, 4.1]

convenience = [4.6, 4.4, 4.8, 4.6, 4.9, 4.3, 4.7, 4.5, 4.7, 4.6, 4.6, 4.4, 4.8, 4.6, 4.9, 4.3, 4.7, 4.5, 4.7, 4.6, 4.6, 4.4, 4.8, 4.6, 4.9, 4.3, 4.7, 4.5, 4.7, 4.6]

# Function to calculate mean
def calculate_mean(data):
    return sum(data) / len(data)

# Function to calculate standard deviation
def calculate_sd(data):
    mean = calculate_mean(data)
    variance = sum((x - mean)**2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

# Calculate means and standard deviations for each category
print("Likert Scale Responses Analysis:")
print("-" * 100)
print("{:<25} {:<15} {:<15} {:<15} {:<15}".format(
    "Category", "Calculated Mean", "Expected Mean", "Calculated SD", "Expected SD"))
print("-" * 100)

# Ease of Use
ease_mean = calculate_mean(ease_of_use)
ease_sd = calculate_sd(ease_of_use)
print("{:<25} {:<15.9f} {:<15.9f} {:<15.10f} {:<15.10f}".format(
    "Ease of Use", ease_mean, 4.233333333, ease_sd, 0.2411657512))

# Engagement and Interest
engagement_mean = calculate_mean(engagement_interest)
engagement_sd = calculate_sd(engagement_interest)
print("{:<25} {:<15.9f} {:<15.9f} {:<15.10f} {:<15.10f}".format(
    "Engagement and Interest", engagement_mean, 4.496666667, engagement_sd, 0.1938419785))

# Improvement in Speaking Skills
speaking_mean = calculate_mean(speaking_skills)
speaking_sd = calculate_sd(speaking_skills)
print("{:<25} {:<15.9f} {:<15.9f} {:<15.10f} {:<15.10f}".format(
    "Speaking Skills", speaking_mean, 4.063333333, speaking_sd, 0.2108821101))

# Personalization of Learning
personalization_mean = calculate_mean(personalization)
personalization_sd = calculate_sd(personalization)
print("{:<25} {:<15.9f} {:<15.9f} {:<15.10f} {:<15.10f}".format(
    "Personalization", personalization_mean, 4.116666667, personalization_sd, 0.1723735585))

# Convenience of Learning
convenience_mean = calculate_mean(convenience)
convenience_sd = calculate_sd(convenience)
print("{:<25} {:<15.9f} {:<15.9f} {:<15.10f} {:<15.10f}".format(
    "Convenience", convenience_mean, 4.593333333, convenience_sd, 0.1964044619))