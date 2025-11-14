#date: 2025-11-14T16:59:16Z
#url: https://api.github.com/gists/cefc0fc42e3b1f15d4068a1cc2417079
#owner: https://api.github.com/users/mattsebastianh

import pandas as pd
import numpy as np

car_eval = pd.read_csv('car_eval_dataset.csv')

# Task Group 1

# 1. Create a table of frequencies of all the cars reviewed by "manufacturer_country". 
# What is the modal category? Which country appears 4th most frequently?

# Note: `.value_counts()` produces a table of frequencies in order, you can reference the
# N-th row of this table to find the Nth most common value in the data. In this case, it's Germany and the US...
print(car_eval.manufacturer_country.value_counts())

# 2. Calculate a table of proportions for countries that appear in "manufacturer_country" in the dataset. 

# Note: Using normalize w. `.value_counts()` normalizes the table of frequencies that `.value_counts() produces by default to a table of proportions. A table of frequencies takes the count of observations, a table of proportions takes the proportion each value represents of the total. 
print(car_eval.manufacturer_country.value_counts(normalize=True))

# Japan            0.228


# Task Group 2

# 3. "buying_cost" is a categorical variable which describes the cost of buying any car in the dataset.
# Note: The `.unique()` method strips all unique values from a column. 

print(car_eval["buying_cost"].unique())

# 4. Create a list of the unique categories in from lowest to highest
# cost in the "buying_cost" variable.

# NOTE: Learner just needs to create their own list and save it for use in 
# subsequent steps
buying_cost_categories = ['low', 'med', 'high', 'vhigh']
print(buying_cost_categories)

# 5. Convert `buying_cost` to type `'category'` using the order created in the previous
# exercise.

# Note: You can convert a field to type category using the function `pandas.Categorical()`. The pandas categorical type allows us to perform numeric operations on categorical data.

# You can also check the column has type category by checking `print(car_eval.buying_cost)` The output out that should note the column type (see below)

car_eval["buying_cost"] = pd.Categorical(
    car_eval["buying_cost"],
    buying_cost_categories,
    ordered=True
)

print(car_eval.buying_cost)

# Name: buying_cost, Length: 1000, dtype: category
# Categories (4, object): [low < med < high < vhigh]

# 6. Calculate the median category of the `buying_cost` variable.

# Note: In Python, you can use `np.median()` to calculate the median value of a numerical series. In this case, you also must access the numerical values of the categories. This can be done with the `.cat.codes` attribute. 


median_category_num = np.median(car_eval['buying_cost'].cat.codes)
print(median_category_num) 

median_category = buying_cost_categories[int(median_category_num)]
print(median_category)

# Task Group 3

# 7. Calculate a table of proportions for 'luggage'.

# Note: Using normalize w. `.value_counts()` normalizes the 
# table of frequencies that `value_counts() produces by default 
# to a table of proportions
print(car_eval.luggage.value_counts(normalize=True))

# 8. Are there any missing values in this column? Replicate the table of proportions from the previous exercise, but do not exclude any missing values from the count.

# Note: Using `.value_counts()`, missing values are removed by default. To keep missing values in the summary you can pass `dropna = False` to `.value_counts()`. If the result using `dropna = False` is the same, then you can conclude there are no missing values. In this case, we see they are the same.

print(car_eval.luggage.value_counts(dropna=False, normalize=True))


# 9. Without passing `normalize = True` to `.value_counts()`, 
# can you replicate the result you got in the previous exercise?

# Note: This method relies on `luggage` having no null values. 
# If a field does have nulls The below is a more robust solution,
# using the `.count()` method excludes NULLs in the denominator just 
# as `.value_counts(normalize=True)` does

print(car_eval.luggage.value_counts()/len(car_eval.luggage))

# Safe alternative if there are Nulls:
car_eval.luggage.value_counts()/car_eval.luggage.count()

# Task Group 4

# 10/11. Find the frequency and proportion of cars that have 5 or more doors. 

# Note: You must first create a series that evaluates to true/false values. 
# Then, by calling .sum() and .mean() on this series the value and count of 
# `True`/1 values are calculated


frequency = (car_eval.doors == '5more').sum()
proportion = (car_eval.doors == '5more').mean()
print(frequency)
print(proportion)