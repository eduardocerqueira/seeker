#date: 2025-04-16T16:37:40Z
#url: https://api.github.com/gists/a6821d3f6bd71d2133ff8731bf01c515
#owner: https://api.github.com/users/tankist52

# Все потенциальные признаки
features = ['latitude', 'longitude', 'elevation', 'temperature', 'cadence',
            'distance_to_previous', 'region', 'season', 'has_building', 'has_railway', 'has_highway', 'time_of_day']

X = df[features]
y = df['Риск']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

