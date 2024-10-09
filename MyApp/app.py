from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib
import jsonify
import time
from datetime import date 

app = Flask(__name__)

# Load the models
spam_model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

sales_model = joblib.load('sales_model.pkl')
car_model = joblib.load('car_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/spam', methods=['GET', 'POST'])
def spam():
    prediction = None
    accuracy = None
    probabilities = None

    if request.method == 'POST':
        message = request.form['message']
        message_tfidf = vectorizer.transform([message])
        prediction = spam_model.predict(message_tfidf)[0]
        accuracy = spam_model.predict_proba(message_tfidf).max() * 100
        
   
        probabilities = {
            'ham': spam_model.predict_proba(message_tfidf)[0][0] * 100,
            'spam': spam_model.predict_proba(message_tfidf)[0][1] * 100
        }
    return render_template('spam.html', prediction=prediction, accuracy=accuracy, probabilities=probabilities)

@app.route('/sales', methods=['GET', 'POST'])
def sales():
    prediction = None 
    if request.method == 'POST':
        try:
            features = [float(request.form[feature]) for feature in ['tv', 'radio', 'newspaper']]
            print("Features for prediction:", features)  
            
            prediction = sales_model.predict([features])[0]
            print("Prediction:", prediction)  

        except Exception as e:
            print("Error during prediction:", e)  

    return render_template('sales.html', prediction=prediction)

@app.route('/car', methods=['GET', 'POST'])
def car():
    car_names = [
        'ritz', 'sx4', 'ciaz', 'wagon r', 'swift', 'vitara brezza',
       's cross', 'alto 800', 'ertiga', 'dzire', 'alto k10', 'ignis',
       '800', 'baleno', 'omni', 'fortuner', 'innova', 'corolla altis',
       'etios cross', 'etios g', 'etios liva', 'corolla', 'etios gd',
       'camry', 'land cruiser', 'Royal Enfield Thunder 500',
       'UM Renegade Mojave', 'KTM RC200', 'Bajaj Dominar 400',
       'Royal Enfield Classic 350', 'KTM RC390', 'Hyosung GT250R',
       'Royal Enfield Thunder 350', 'KTM 390 Duke ',
       'Mahindra Mojo XT300', 'Bajaj Pulsar RS200',
       'Royal Enfield Bullet 350', 'Royal Enfield Classic 500',
       'Bajaj Avenger 220', 'Bajaj Avenger 150', 'Honda CB Hornet 160R',
       'Yamaha FZ S V 2.0', 'Yamaha FZ 16', 'TVS Apache RTR 160',
       'Bajaj Pulsar 150', 'Honda CBR 150', 'Hero Extreme',
       'Bajaj Avenger 220 dtsi', 'Bajaj Avenger 150 street',
       'Yamaha FZ  v 2.0', 'Bajaj Pulsar  NS 200', 'Bajaj Pulsar 220 F',
       'TVS Apache RTR 180', 'Hero Passion X pro', 'Bajaj Pulsar NS 200',
       'Yamaha Fazer ', 'Honda Activa 4G', 'TVS Sport ',
       'Honda Dream Yuga ', 'Bajaj Avenger Street 220',
       'Hero Splender iSmart', 'Activa 3g', 'Hero Passion Pro',
       'Honda CB Trigger', 'Yamaha FZ S ', 'Bajaj Pulsar 135 LS',
       'Activa 4g', 'Honda CB Unicorn', 'Hero Honda CBZ extreme',
       'Honda Karizma', 'Honda Activa 125', 'TVS Jupyter',
       'Hero Honda Passion Pro', 'Hero Splender Plus', 'Honda CB Shine',
       'Bajaj Discover 100', 'Suzuki Access 125', 'TVS Wego',
       'Honda CB twister', 'Hero Glamour', 'Hero Super Splendor',
       'Bajaj Discover 125', 'Hero Hunk', 'Hero  Ignitor Disc',
       'Hero  CBZ Xtreme', 'Bajaj  ct 100', 'i20', 'grand i10', 'i10',
       'eon', 'xcent', 'elantra', 'creta', 'verna', 'city', 'brio',
       'amaze', 'jazz'
    ]

    prediction = None
    try:
        if request.method == 'POST':
            
            car_name = request.form['Car_Name']
            year = int(request.form['Year'])
            present_price = float(request.form['Present_Price'])
            driven_kms = float(request.form['Driven_kms'])
            fuel_type = request.form['Fuel_Type']
            selling_type = request.form['Selling_Type']
            transmission = request.form['Transmission']
            owner = int(request.form['Owner'])

            
            current_year = date.today().year
            car_age = current_year - year

            
            features = [car_name, year, present_price, driven_kms, fuel_type, selling_type, transmission, owner]

            
            scaler = joblib.load('scaler.pkl')  


            encoders = joblib.load('label_encoder.pkl')  
    
            import pandas as pd 
            
            
            car_name_encoded = encoders['Car_Name'].get(car_name, -1)
            fuel_type_encoded = encoders['Fuel_Type'].get(fuel_type, -1)  
            selling_type_encoded = encoders['Selling_type'].get(selling_type, -1)
            transmission_encoded = encoders['Transmission'].get(transmission, -1)
            
            df = pd.DataFrame({
                    'Car_Name': [car_name_encoded],
                    'Present_Price': [present_price],
                    'Driven_kms': driven_kms,
                    'Fuel_Type': [fuel_type_encoded],
                    'Selling_type': [selling_type_encoded],
                    'Transmission': [transmission_encoded],
                    'Owner': [owner],
                    'Car_Age': [car_age]
                })
            
            X_scaled = scaler.transform(df)
            
            prediction = car_model.predict(X_scaled)

    except Exception as e:
            print("Error:", str(e))  
    
    return render_template('car.html', prediction=prediction, car_names=car_names)

if __name__ == '__main__':
    app.run(debug=True)
