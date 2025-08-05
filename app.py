import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from pymongo import MongoClient
from bson import ObjectId
import os

app = Flask(__name__)
CORS(app)

# --- MongoDB Setup ---
MONGO_URI = os.environ.get("MONGO_URI") # Vercel environment variable se URI lein
client = MongoClient(MONGO_URI)
db = client['hospital_db']
appointments_collection = db['appointments']

# --- Global variables ---
model = None
le = None
symptom_list = []
symptom_description_df = None
symptom_precaution_df = None

def load_model_and_data():
    """
    Saved model aur data files ko load karein.
    """
    global model, le, symptom_list, symptom_description_df, symptom_precaution_df
    
    # Load the pre-trained model and lists
    model = joblib.load('disease_model.pkl')
    le = joblib.load('label_encoder.pkl')
    symptom_list = joblib.load('symptom_list.pkl')
    
    # Load the description and precaution CSVs
    symptom_description_df = pd.read_csv('symptom_Description.csv')
    symptom_precaution_df = pd.read_csv('symptom_precaution.csv')
    
    print("Model and data loaded successfully!")

# Server start hone par model load karein
load_model_and_data()

# Baaki ke routes (predict, symptoms, etc.) pehle jaise hi rahenge...
@app.route('/predict', methods=['POST'])
def predict():
    # ... (Is function mein koi badlaav nahi hai) ...
    if model is None or le is None: return jsonify({'error': 'Model not loaded yet.'}), 503
    try:
        data = request.get_json()
        user_symptoms = data.get('symptoms', [])
        if not user_symptoms: return jsonify({'error': 'No symptoms provided'}), 400
        input_vector = pd.DataFrame(0, index=[0], columns=symptom_list)
        for symptom in user_symptoms:
            cleaned_symptom = symptom.strip().replace('_', ' ')
            if cleaned_symptom in input_vector.columns:
                input_vector.loc[0, cleaned_symptom] = 1
        prediction_numeric = model.predict(input_vector)[0]
        prediction = le.inverse_transform([prediction_numeric])[0]
        description = symptom_description_df[symptom_description_df['Disease'] == prediction]['Description'].values[0]
        precautions_df = symptom_precaution_df[symptom_precaution_df['Disease'] == prediction]
        precautions = []
        if not precautions_df.empty:
            precautions = [p for p in precautions_df.iloc[:, 1:].values[0] if pd.notna(p)]
        return jsonify({'predicted_disease': prediction, 'description': description, 'precautions': precautions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    if not symptom_list: return jsonify({'error': 'Symptoms not available yet.'}), 503
    return jsonify({'symptoms': symptom_list})

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    try:
        data = request.get_json()
        name, phone, date = data.get('name'), data.get('phone'), data.get('date')
        if not name or not phone or not date: return jsonify({'error': 'Please fill all the fields.'}), 400
        appointment_data = {"name": name, "phone": phone, "date": date, "status": "Pending"}
        appointments_collection.insert_one(appointment_data)
        return jsonify({'message': f'Appointment for {name} on {date} has been requested successfully!'}), 200
    except Exception as e:
        return jsonify({'error': 'Could not save appointment.'}), 500

@app.route('/appointments', methods=['GET'])
def get_appointments():
    try:
        all_appointments = list(appointments_collection.find({}))
        for appointment in all_appointments:
            appointment['_id'] = str(appointment['_id'])
        return jsonify(all_appointments), 200
    except Exception as e:
        return jsonify({'error': 'Could not fetch appointments.'}), 500

@app.route('/update_status', methods=['POST'])
def update_status():
    try:
        data = request.get_json()
        appointment_id = data.get('id')
        new_status = data.get('status')
        if not appointment_id or not new_status:
            return jsonify({'error': 'Missing appointment ID or new status.'}), 400
        appointments_collection.update_one(
            {'_id': ObjectId(appointment_id)},
            {'$set': {'status': new_status}}
        )
        return jsonify({'message': 'Status updated successfully!'}), 200
    except Exception as e:
        return jsonify({'error': 'Could not update status.'}), 500

# Vercel ke liye `app.run()` ki zaroorat nahi hoti
