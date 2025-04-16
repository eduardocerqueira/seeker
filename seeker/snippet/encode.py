#date: 2025-04-16T16:37:40Z
#url: https://api.github.com/gists/a6821d3f6bd71d2133ff8731bf01c515
#owner: https://api.github.com/users/tankist52

from sklearn.preprocessing import LabelEncoder

# Категориальные признаки
categorical_cols = ['region', 'season', 'time_of_day']

# Бинарные признаки: преобразуем True/False в 0/1
bool_cols = ['has_building', 'has_wood', 'has_river', 'has_railway', 'has_highway']
df[bool_cols] = df[bool_cols].astype(int)

# Кодировка категориальных признаков
le_dict = {}
for col in categorical_cols + ['Риск']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # Сохраняем энкодеры, если понадобится декодировать позже

