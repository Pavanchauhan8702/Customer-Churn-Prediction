import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Define a function to preprocess the data
def preprocess_data(df):
    # Replace missing values with mean
    df.fillna(df.mean(), inplace=True)
    # Convert categorical variables to numerical using one-hot encoding
    df = pd.get_dummies(df, columns=['gender', 'education', 'marital_status', 'payment_method'])
    return df

# Load the data
df = pd.read_csv('train.csv')

# Preprocess the data
df = preprocess_data(df)

# Split the data into X and y
X = df.drop(['churn'], axis=1)
y = df['churn']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_scaled, y)

# Define a function to make predictions using the model
def predict_churn(gender, age, education, marital_status, income, payment_method, monthly_charge, data_usage, total_calls_made):
    # Convert input data to a dataframe
    input_data = pd.DataFrame({'gender': gender,
                                'age': age,
                                'education': education,
                                'marital_status': marital_status,
                                'income': income,
                                'payment_method': payment_method,
                                'monthly_charge': monthly_charge,
                                'data_usage': data_usage,
                                'total_calls_made': total_calls_made}, index=[0])

    # Preprocess the input data
    input_data = preprocess_data(input_data)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction using the trained model
    prediction = logreg.predict(input_data_scaled)[0]

    return prediction

# Create the Streamlit app
def main():
    st.title('Telecom Customer Churn Prediction')

    # Add input fields for user to enter data
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 18, 100)
    education = st.selectbox('Education', ['High School or Below', 'College', 'Bachelor', 'Master or Above'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    income = st.slider('Income', 0, 100000, 50000)
    payment_method = st.selectbox('Payment Method', ['Credit Card', 'Bank Transfer', 'Mailed Check'])
    monthly_charge = st.slider('Monthly Charge', 0, 500, 100)
    data_usage = st.slider('Data Usage', 0, 50, 10)
    total_calls_made = st.slider('Total Calls Made', 0, 20, 5)

    # Make a prediction using the predict_churn function
    prediction = predict_churn(gender, age, education, marital_status, income, payment_method, monthly_charge, data_usage, total_calls_made)

    # Display the prediction
    st.subheader('Prediction')
    if prediction == 0:
        st.write('Customer is not likely to churn.')
    else:
        st.write('Customer is likely to churn.')

# Run the Streamlit app
if _name_== '_main_':
    main()