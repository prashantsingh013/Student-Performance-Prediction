import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Define the file path
# Assuming the script runs from the same directory where the csv is.
file_path = 'student_data.csv'

# Check if file exists
if not os.path.exists(file_path):
    # Try to find it relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'student_data.csv')

def load_data(path):
    """Load the dataset from CSV file."""
    try:
        df = pd.read_csv(path)
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        return None

def preprocess_data(df):
    """Preprocess the dataset (handle missing values, etc.)."""
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Missing values found. Filling with mean.")
        df.fillna(df.mean(), inplace=True)
    else:
        print("No missing values found.")
    
    return df

def feature_engineering(df):
    """Create new features."""
    # Example: Total Study Effort (Interaction term)
    # The logic is that high attendance combined with high study hours likely leads to better results.
    df['Study_Effort'] = df['Attendance_Percentage'] * df['Study_Hours_Per_Week']
    print("Feature engineering completed: Added 'Study_Effort'.")
    return df

def train_model(X, y):
    """Train a Linear Regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using RMSE and R2 Score."""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (Accuracy): {r2:.2f}")
    
    return y_pred

def main():
    # 1. Load Data
    df = load_data(file_path)
    if df is None:
        return

    # 2. Preprocessing
    df = preprocess_data(df)

    # 3. Feature Engineering
    df = feature_engineering(df)

    # Prepare features and target
    # Target: Final_Grade
    # Features: Attendance_Percentage, Assignment_Scores, Study_Hours_Per_Week, Previous_Exam_Results, Study_Effort
    X = df[['Attendance_Percentage', 'Assignment_Scores', 'Study_Hours_Per_Week', 'Previous_Exam_Results', 'Study_Effort']]
    y = df['Final_Grade']

    # 4. Train Model
    model, X_test, y_test = train_model(X, y)

    # 5. Evaluate
    y_pred = evaluate_model(model, X_test, y_test)

    # Print a few predictions vs actuals
    print("\nSample Predictions:")
    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(comparison.head())

    # Get coefficients to see feature importance
    print("\nFeature Importance (Coefficients):")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")

if __name__ == "__main__":
    main()
