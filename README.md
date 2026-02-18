🎓 Student Performance Prediction using Machine Learning
📌 Overview

This project focuses on predicting students’ final grades using machine learning techniques. Academic and behavioral factors such as attendance percentage, assignment scores, weekly study hours, and previous exam results are used to build a predictive model. The goal is to analyze how these factors influence student performance and generate accurate grade predictions.

🚀 Features

Data loading from CSV file

Data preprocessing and missing value handling

Feature engineering for improved prediction accuracy

Model training using Linear Regression

Model evaluation using RMSE and R² score

Analysis of feature importance

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

📂 Project Structure
📁 Student-Performance-Prediction
│── train_model.py
│── student_data.csv
│── README.md

📊 Dataset Description

The dataset contains the following features:

Feature Name	Description
Attendance_Percentage	Student attendance percentage
Assignment_Scores	Average assignment score
Study_Hours_Per_Week	Weekly study hours
Previous_Exam_Results	Previous exam performance
Final_Grade	Target variable (predicted output)
⚙️ How It Works

Loads student data from a CSV file

Cleans the data and handles missing values

Creates an additional feature to represent combined study effort

Splits the dataset into training and testing sets

Trains a Linear Regression model

Evaluates the model using RMSE and R² score

Displays predictions and feature importance

▶️ How to Run the Project
pip install pandas numpy scikit-learn
python train_model.py

📈 Model Evaluation

RMSE (Root Mean Squared Error) is used to measure prediction error

R² Score is used to evaluate how well the model fits the data

🔍 Results

The model provides predicted final grades along with performance metrics. Feature coefficients help understand which factors contribute most to student performance.

🎯 Use Case

Academic performance analysis

Early identification of students at risk

Educational data analysis projects

🤝 Contributing

Contributions are welcome. Feel free to fork this repository and submit a pull request.
