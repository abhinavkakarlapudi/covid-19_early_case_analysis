import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the Dataset
# Note: Ensure the 'PatientInfo.csv' file is in the same directory.
try:
    df = pd.read_csv('PatientInfo.csv')
except FileNotFoundError:
    print("Dataset not found. Please ensure the CSV is in the correct directory.")
    df = pd.DataFrame() # Placeholder

if not df.empty:
    # 2. Data Cleaning & Preprocessing
    # Converting dates to datetime objects
    df['confirmed_date'] = pd.to_datetime(df['confirmed_date'])
    df['released_date'] = pd.to_datetime(df['released_date'])

    # Feature Engineering: Calculate Recovery Duration
    # Recovery duration is only applicable for 'released' patients
    df['recovery_duration'] = (df['released_date'] - df['confirmed_date']).dt.days

    # Calculate Age from birth_year
    current_year = 2020 # Context of the dataset
    df['age'] = current_year - df['birth_year']

    # 3. Descriptive Statistics
    print("--- Dataset Summary ---")
    print(df[['age', 'contact_number', 'recovery_duration']].describe())

    # 4. Visualizations
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 10))

    # Gender Distribution
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='sex', palette='viridis')
    plt.title('Gender Distribution of Infected Patients')

    # Age Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['age'].dropna(), bins=20, kde=True, color='salmon')
    plt.title('Age Distribution of Patients')

    # Regional Case Concentration (Top 10)
    plt.subplot(2, 2, 3)
    df['region'].value_counts().head(10).plot(kind='bar', color='teal')
    plt.title('Top 10 Impacted Regions')
    plt.xticks(rotation=45)

    # Infection Reasons
    plt.subplot(2, 2, 4)
    df['infection_reason'].value_counts().head(5).plot(kind='pie', autopct='%1.1f%%')
    plt.title('Primary Infection Reasons')

    plt.tight_layout()
    plt.show()

    # 5. Linear Regression: Predict Recovery Time
    # Filtering data: We need patients who have a recorded recovery duration and age
    reg_data = df.dropna(subset=['age', 'contact_number', 'recovery_duration'])
    reg_data = reg_data[reg_data['recovery_duration'] >= 0] # Ensure valid data

    if not reg_data.empty:
        X = reg_data[['age', 'contact_number']] # Independent variables
        y = reg_data['recovery_duration']       # Dependent variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions and Evaluation
        y_pred = model.predict(X_test)
        
        print("\n--- Linear Regression Results ---")
        print(f"R-squared Score: {r2_score(y_test, y_pred):.4f}")
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
        print(f"Coefficients: Age={model.coef_[0]:.2f}, Contact Number={model.coef_[1]:.2f}")
    else:
        print("\nInsufficient data for Linear Regression modeling.")
        