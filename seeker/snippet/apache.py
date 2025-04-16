#date: 2025-04-16T16:37:40Z
#url: https://api.github.com/gists/a6821d3f6bd71d2133ff8731bf01c515
#owner: https://api.github.com/users/tankist52

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
import os

# Параметры подключения к базе данных (изменён для Ubuntu)
DATABASE_URI = "postgresql://postgres:12345@localhost:5432/aaa"  # Изменено на localhost, если PostgreSQL работает на хосте Ubuntu
MODEL_PATH = '/home/airflow/risk_model.pkl'  # Путь для сохранения модели на Ubuntu

def fetch_full_data(**kwargs):
    engine = create_engine(DATABASE_URI)
    query = "SELECT * FROM aaa"  # Название таблицы в БД
    df = pd.read_sql(query, engine)
    kwargs['ti'].xcom_push(key='full_data', value=df)
    print(f"Загружено {len(df)} строк.")

def train_risk_model(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids='fetch_full_data', key='full_data')

    # Преобразование булевых значений
    bool_cols = ['has_building', 'has_wood', 'has_river', 'has_railway', 'has_highway']
    df[bool_cols] = df[bool_cols].astype(int)

    # Категориальные признаки
    categorical_cols = ['Region', 'season', 'time_of_day']
    le_dict = {}
    for col in categorical_cols + ['Риск']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Признаки и цель
    features = ['latitude', 'longitude', 'elevation', 'temperature', 'heart_rate', 'cadence',
                'distance_to_previous', 'Region', 'season', 'time_of_day',
                'has_building', 'has_railway', 'has_highway']
    X = df[features]
    y = df['Риск']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Точность модели:", acc)
    print("Отчёт классификации:\n", classification_report(y_test, y_pred, target_names=le_dict['Риск'].classes_))

    # Сохраняем модель и энкодеры
    joblib.dump({
        'model': rf,
        'label_encoders': le_dict,
        'features': features
    }, MODEL_PATH)

    print(f"Модель и энкодеры сохранены по пути: {MODEL_PATH}")

# DAG
with DAG(
        'risk_classification_training',
        default_args={'retries': 1},
        description='Обучение модели классификации риска для треков',
        schedule_interval='@daily',
        start_date=datetime(2023, 1, 1),
        catchup=False,
) as dag:

    fetch_data_task = PythonOperator(
        task_id='fetch_full_data',
        python_callable=fetch_full_data,
        provide_context=True
    )

    train_model_task = PythonOperator(
        task_id='train_risk_model',
        python_callable=train_risk_model,
        provide_context=True
    )

    fetch_data_task >> train_model_task
