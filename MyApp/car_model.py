import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from datetime import date

def load_data(file_path):
    car_df = pd.read_csv(file_path)
    return car_df


def preprocess_data(df):
    current_year = date.today().year
    df['Car_Age'] = current_year - df['Year']

    # Initialize a dictionary to store label encoders for each categorical column
    encoders = {}

    # List of categorical columns to encode
    categorical_columns = [0, 5, 6, 7]  # Update these indices based on your actual column indices

    # Apply LabelEncoder to each categorical column and store the encoder mapping in the dictionary
    for column in categorical_columns:
        le = LabelEncoder()
        df.iloc[:, column] = le.fit_transform(df.iloc[:, column])
        
        # Save the encoder mapping as a dictionary
        column_name = df.columns[column]
        encoders[column_name] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Drop the 'Selling_Price' and 'Year' columns
    X = df.drop(['Selling_Price', 'Year'], axis=1)
    y = df['Selling_Price']

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler and the label encoders
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
    joblib.dump(encoders, 'label_encoder.pkl')  # Save all encoders in a single file

    return X_scaled, y

def train_car_price_model():
    file_path = 'car_data.csv'
    df = load_data(file_path)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model using joblib
    joblib.dump(model, 'car_model.pkl')

if __name__ == '__main__':
    train_car_price_model()
    print("Car Model saved!")
