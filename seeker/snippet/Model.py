#date: 2023-05-19T16:57:02Z
#url: https://api.github.com/gists/5a7515eb266f48bb281ad0b9ce135295
#owner: https://api.github.com/users/HarshaVardhan3002

import os
import pandas as pd
from multiprocessing import Pool
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pickle

def read_data_from_file(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=',')
        if not df.empty:
            return df
        else:
            print(f"Empty file: {file_path}. Skipping...")
    except pd.errors.EmptyDataError:
        print(f"Empty file: {file_path}. Skipping...")
    return pd.DataFrame()

def read_data_from_files(directory):
    file_paths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            file_paths.append(file_path)

    with Pool() as pool:
        results = pool.map(read_data_from_file, file_paths)
        data_frames = [df for df in results if not df.empty]
        total_file_count = len(data_frames)

    if data_frames:
        data = pd.concat(data_frames, ignore_index=True)
    else:
        data = pd.DataFrame()

    return data, total_file_count, len(results) - total_file_count

if __name__ == '__main__':
    stocks_directory = r"Dataset location"

    stocks_data, file_count, skipped_files = read_data_from_files(stocks_directory)

    # Additional code for model training...
    if not stocks_data.empty:
        stocks_data['Date'] = pd.to_datetime(stocks_data['Date'])

        # Preprocess the data
        stocks_data['Date'] = pd.to_datetime(stocks_data['Date'])
        stocks_data.set_index('Date', inplace=True)
        stocks_data.sort_index(inplace=True)
        stocks_data.dropna(inplace=True)

        # Split the data into input features (X) and target variable (y)
        X = stocks_data.drop('Close', axis=1)
        y = stocks_data['Close']

        # Handle NaN values in the input features
        X.fillna(0, inplace=True)

        # Scale the input features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        score = model.score(X_test, y_test)
        print("Model R-squared score:", score)

        # Save the model as a pickle file
        model_path = r"Location to save model"
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

    print("Total files:", file_count)
    print("Empty or skipped files:", skipped_files)

else:
    print("No data available.")
