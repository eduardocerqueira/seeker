#date: 2025-04-16T16:37:40Z
#url: https://api.github.com/gists/a6821d3f6bd71d2133ff8731bf01c515
#owner: https://api.github.com/users/tankist52

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Загрузка данных
df1 = df

# Преобразуем bool в int
bool_cols = df1.select_dtypes(include=bool).columns
df1[bool_cols] = df1[bool_cols].astype(int)

# Кодируем категориальные признаки
df_encoded = pd.get_dummies(df1, columns=["region", "season", "time_of_day"])

# Удаляем ненужные столбцы
drop_cols = ["Риск", "Эвакуация", "time", "track_number"]
X = df_encoded.drop(columns=drop_cols)

# Модели случайного леса для "Риск" и "Эвакуация"
# Предполагаем, что у вас есть целевые переменные в df1
y_risk = df1["Риск"]
y_evac = df1["Эвакуация"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_risk_train, y_risk_test = train_test_split(X, y_risk, test_size=0.3, random_state=42)
X_train, X_test, y_evac_train, y_evac_test = train_test_split(X, y_evac, test_size=0.3, random_state=42)

# Обучение моделей случайного леса
risk_model_rf = RandomForestClassifier(random_state=42)
evac_model_rf = RandomForestClassifier(random_state=42)

risk_model_rf.fit(X_train, y_risk_train)
evac_model_rf.fit(X_train, y_evac_train)

# Функция генерации данных для одного года с использованием новых моделей
def generate_year_data(X, original_df, year, risk_model, evac_model):
    new_X = X.copy()
    # Изменим температуру в диапазоне ±5 градусов
    new_X["temperature"] = new_X["temperature"].apply(lambda x: x + np.random.uniform(-5, 5))
    new_df = new_X.copy()
    new_df["year"] = year
    # Предсказания с использованием новых моделей
    new_df["Риск"] = risk_model.predict(new_X)
    new_df["Эвакуация"] = evac_model.predict(new_X)
    return new_df

# Генерация данных на 10 лет вперёд
all_years_data = []
for i in range(1, 11):
    year = 2024 + i
    year_data = generate_year_data(X, df1, year, risk_model_rf, evac_model_rf)
    all_years_data.append(year_data)

# Объединяем все в один DataFrame
forecast_df = pd.concat(all_years_data, ignore_index=True)

# Визуализация среднего по кластерам
def plot_cluster_trend(df, cluster_name):
    plt.figure(figsize=(10, 5))
    cluster_avg = df.groupby("year")[cluster_name].value_counts(normalize=True).unstack()
    cluster_avg.plot(kind='line', marker='o', ax=plt.gca())
    plt.title(f"Прогноз изменения кластера '{cluster_name}' на 10 лет")
    plt.ylabel("Доля значений")
    plt.xlabel("Год")
    plt.grid(True)
    plt.legend(title=cluster_name)
    plt.tight_layout()
    plt.show()

# Визуализируем прогнозы
plot_cluster_trend(forecast_df, "Риск")
plot_cluster_trend(forecast_df, "Эвакуация")