# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib 
from pymongo import MongoClient
from bson import ObjectId 
from flask import render_template
# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) 

# --- MongoDB Setup ---
# APNI MONGODB ATLAS CONNECTION STRING YAHAAN PASTE KAREIN
MONGO_URI = "mongodb+srv://kunal123:UuM9Ttt62Zqw3hS1@cluster0.mm3qhiy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" 
client = MongoClient(MONGO_URI)
db = client['hospital_db'] 
appointments_collection = db['appointments'] 

# --- Global variables for the model and encoders ---
model = None
le = None
symptom_list = []
symptom_description_df = None
symptom_precaution_df = None

def train_model():
    """
    This function loads the data, preprocesses it, trains the model,
    and prepares all necessary components for prediction.
    It's called once when the server starts.
    """
    global model, le, symptom_list, symptom_description_df, symptom_precaution_df
    try:
        # File paths for locally extracted CSV files
        training_df = pd.read_csv('dataset.csv')
        symptom_description_df = pd.read_csv('symptom_Description.csv')
        symptom_precaution_df = pd.read_csv('symptom_precaution.csv')
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please ensure 'dataset.csv', 'symptom_Description.csv', and 'symptom_precaution.csv' are in the same directory.")
        return

    symptom_columns = [f'Symptom_{i}' for i in range(1, 18)]
    all_symptoms = set()
    for col in symptom_columns:
        training_df[col] = training_df[col].astype(str).str.strip().str.replace('_', ' ')
        unique_in_col = training_df[col][training_df[col] != 'nan'].unique()
        all_symptoms.update(unique_in_col)
    symptom_list = sorted(list(all_symptoms))
    X = pd.DataFrame(0, index=training_df.index, columns=symptom_list)
    for index, row in training_df.iterrows():
        for col in symptom_columns:
            symptom = row[col]
            if symptom != 'nan' and symptom in symptom_list:
                X.loc[index, symptom] = 1
    le = LabelEncoder()
    y = le.fit_transform(training_df['Disease'])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained and ready!")

@app.route('/predict', methods=['POST'])
def predict():
    """ API endpoint for disease prediction. """
    if model is None or le is None: return jsonify({'error': 'Model not trained yet.'}), 503
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
    """ API endpoint to get all symptoms. """
    if not symptom_list: return jsonify({'error': 'Symptoms not available yet.'}), 503
    return jsonify({'symptoms': symptom_list})

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    """ API endpoint to handle appointment booking form submissions and save to MongoDB. """
    try:
        data = request.get_json()
        name, phone, date = data.get('name'), data.get('phone'), data.get('date')
        if not name or not phone or not date: return jsonify({'error': 'Please fill all the fields.'}), 400
        appointment_data = {"name": name, "phone": phone, "date": date, "status": "Pending"}
        appointments_collection.insert_one(appointment_data)
        print(f"Successfully saved appointment for {name} on {date}")
        return jsonify({'message': f'Appointment for {name} on {date} has been requested successfully!'}), 200
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return jsonify({'error': 'Could not save appointment.'}), 500

# --- Admin Dashboard Endpoints ---

@app.route('/appointments', methods=['GET'])
def get_appointments():
    """
    API endpoint to fetch all appointments from the database for the admin dashboard.
    """
    try:
        all_appointments = list(appointments_collection.find({}))
        for appointment in all_appointments:
            appointment['_id'] = str(appointment['_id'])
        return jsonify(all_appointments), 200
    except Exception as e:
        print(f"Error fetching appointments: {e}")
        return jsonify({'error': 'Could not fetch appointments.'}), 500

@app.route('/update_status', methods=['POST'])
def update_status():
    """
    API endpoint to update the status of an appointment.
    """
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
        print(f"Updated status for {appointment_id} to {new_status}")
        return jsonify({'message': 'Status updated successfully!'}), 200
    except Exception as e:
        print(f"Error updating status: {e}")
        return jsonify({'error': 'Could not update status.'}), 500

import os

# --- Main Application Execution ---


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/admin')
def admin():
    return render_template('admin.html')


if __name__ == '__main__':
    train_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
