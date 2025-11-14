#date: 2025-11-14T16:59:16Z
#url: https://api.github.com/gists/cefc0fc42e3b1f15d4068a1cc2417079
#owner: https://api.github.com/users/mattsebastianh

import pandas as pd
import numpy as np

car_eval = pd.read_csv('car_eval_dataset.csv')
print(car_eval.head())

print(car_eval.info())
print(car_eval.head())
print("Is there any missing values?", car_eval.isna().sum().any())

## Part I: Summarizing Manufacturing Country
# 1. calculate freqency
print(car_eval['manufacturer_country'].value_counts(normalize=False))
# comment: modal category is "Japan"; "United States" stsads on 4th place

# 2. calculate proportion
print(car_eval['manufacturer_country'].value_counts(normalize=True))
# comment: Japan manufacturing stands in first place with 22.8% of market hold

## Part II: Summarizing Buying Costs
print(car_eval.buying_cost.unique())
print(car_eval.buying_cost.value_counts())  # ordinal categorical variables

# create ordinal categories list
buying_cost_categories = ['low', 'med', 'high', 'vhigh']

# convert to categorical variables
car_eval['buying_cost'] = pd.Categorical(car_eval['buying_cost'], buying_cost_categories, ordered=True)
print(car_eval['buying_cost'].unique())
# calculate median category
median_cat_idx = np.median(car_eval['buying_cost'].cat.codes)
print(median_cat_idx)

# return category object with median index
median_category = buying_cost_categories[int(median_cat_idx)]
print("Median category:", median_category)

## Part III: Sumarrize Luggage Capacity
print(car_eval['luggage'].unique())
print(car_eval['luggage'].value_counts())

# table of proportions
print(car_eval['luggage'].value_counts(normalize=True))
print(car_eval['luggage'].value_counts(normalize=True, dropna=False))
# comment: there is no missing values on this category

# replicate the result with manual calculation
print(car_eval['luggage'].value_counts()/len(car_eval['luggage']))

# ordinal cat list
luggage_categories = ['small', 'med', 'big']

# convert to categorical var
car_eval['luggage'] = pd.Categorical(car_eval['luggage'], luggage_categories, ordered=True)
print(car_eval['luggage'].unique())
print(car_eval['luggage'].cat.codes.unique())

## Part IV: Summarizing Passenger Capacity
print(car_eval['doors'].value_counts())
# filter the category by more than 5 doors
more_5door_freq = (car_eval['doors'] == '5more').sum()
print("Number of vehicles with more than 5 doors:", more_5door_freq)
# filter the category by more than 5 doors
more_5door_proportion = (car_eval['doors'] == '5more').mean()
more_5door_proportion = np.mean(car_eval['doors'] == '5more')
print("Proportion of vehicles with more than 5 doors:", more_5door_proportion)
