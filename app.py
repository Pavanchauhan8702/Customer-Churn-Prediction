import pickle
from flask import Flask, jsonify, request,app,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
#load the model
model =pickle.load(open('lr_model.pkl','rb'))

# Define API endpoints
@app.route('/')
def home():
 return render_template('home.html')

# Define API endpoints
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    gender = data['gender']
    SeniorCitizen = data['SeniorCitizen']
    Partner = data['Partner']
    Dependents = data['Dependents']
    tenure = data['tenure']
    PhoneService = data['PhoneService']
    MultipleLines = data['MultipleLines']
    InternetService = data['InternetService']
    OnlineSecurity = data['OnlineSecurity']
    OnlineBackup = data['OnlineBackup']
    DeviceProtection = data['DeviceProtection']
    TechSupport = data['TechSupport']
    StreamingTV = data['StreamingTV']
    StreamingMovies = data['StreamingMovies']
    Contract = data['Contract']
    PaperlessBilling = data['PaperlessBilling']
    PaymentMethod = data['PaymentMethod']
    MonthlyCharges = data['MonthlyCharges']
    TotalCharges = data['TotalCharges']

 # Preprocess input data
    input_data = pd.DataFrame({'gender': [gender], 'SeniorCitizen': [SeniorCitizen], 'Partner': [Partner], 'Dependents': [Dependents], 'tenure': [tenure], 'PhoneService': [PhoneService], 'MultipleLines': [MultipleLines], 'InternetService': [InternetService], 'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': [OnlineBackup], 'DeviceProtection': [DeviceProtection], 'TechSupport': [TechSupport], 'StreamingTV': [StreamingTV], 'StreamingMovies': [StreamingMovies], 'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling], 'PaymentMethod': [PaymentMethod], 'MonthlyCharges': [MonthlyCharges], 'TotalCharges': [TotalCharges]})
    input_data['TotalCharges'] = pd.to_numeric(input_data['TotalCharges'], errors='coerce')
    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)

 # Make predictions
    lr_pred = lr.predict(input_data)[0]

# Return results
    result = {'Logistic Regression': lr_pred}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
