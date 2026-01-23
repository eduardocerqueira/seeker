#date: 2026-01-23T17:04:19Z
#url: https://api.github.com/gists/f195cc06befeb1bd1f4ab5aa64408658
#owner: https://api.github.com/users/JClarkson2026

import pandas as pd # Importing pandas library for data manipulation

my_collection = { # Sample data representing car crash costs
    'id': [1, 2, 3, 4, 5],
    'name': [
        '2018 Honda Civic – rear-end collision',
        '2020 Ford F-150 – side impact',
        '2016 Toyota Camry – ice skid',
        '2022 Tesla Model 3 – parking lot crash',
        '2015 BMW 328i – highway collision'
    ],
    'quantity': [
        2,  # number of vehicles involved
        2,
        1,
        1,
        3
    ],
    'price': [
        4200.75,
        18500.00,
        2300.40,
        1800.00,
        26500.90
    ],
    'date': [
        '2026-01-01',
        '2026-01-05',
        '2026-01-10',
        '2026-01-12',
        '2026-01-15'
    ]
}

df = pd.DataFrame(my_collection) # Creating a DataFrame from the sample data
df.to_csv('Car_Crash_Cost.csv', index=False) # Saving the DataFrame to a CSV file without the index