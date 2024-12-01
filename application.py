from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Cleaned_Car_data.csv')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Simplified mappings for demonstration
    company_mapping = {
        'Hyundai': 0, 'Mahindra': 1, 'Ford': 2, 'Maruti': 3,
        'Skoda': 4, 'Audi': 5, 'Toyota': 6
    }
    fuel_type_mapping = {'Petrol': 0, 'Diesel': 1, 'LPG': 2}
    name_mapping = {
        'Hyundai Santro Xing': 0,
        'Mahindra Jeep CL550': 1,
        'Hyundai Grand i10': 2,
        'Ford EcoSport Titanium': 3,
        'Maruti Suzuki Alto': 4,
        'Ford Figo': 5,
    }

    # Collect inputs
    company = company_mapping.get(request.form.get('company'), -1)
    car_model = name_mapping.get(request.form.get('car_models'), -1)
    year = int(request.form.get('year'))
    fuel_type = fuel_type_mapping.get(request.form.get('fuel_type'), -1)
    driven = int(request.form.get('kilo_driven'))

    # Ensure valid inputs
    if company == -1 or car_model == -1 or fuel_type == -1:
        return "Invalid input. Please select valid options."

    # Prepare input for prediction as a raw NumPy array
    input_data = np.array([car_model, company, year, driven, fuel_type]).reshape(1, -1)

    # Debug: Print the input data being passed to the model
    print(f"Input Data for Prediction: {input_data}")

    # Make prediction
    try:
        prediction = model.predict(input_data)
        print(f"Prediction: {prediction}")
        return f"Predicted Price: ₹{np.round(prediction[0], 2)}"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error: {str(e)}"



if __name__=='__main__':
    app.run()