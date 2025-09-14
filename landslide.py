from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Get the current folder path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create the Flask app, specifying where templates and static files are
app = Flask(__name__, template_folder=current_dir, static_folder=current_dir)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load both models
with open('model1.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('model2.pkl', 'rb') as file:
    dt_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  # Now looks in the current folder

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    data = np.array([features])
    data_scaled = scaler.transform(data)

    rf_prediction = rf_model.predict(data_scaled)[0]
    dt_prediction = dt_model.predict(data_scaled)[0]

    # Map numeric output to human-readable labels
    result_map = {0: "No Landslide", 1: "Landslide"}

    rf_result = result_map.get(rf_prediction, "Unknown")
    dt_result = result_map.get(dt_prediction, "Unknown")

    return render_template('index.html', 
                       rf_result=rf_result, 
                       dt_result=dt_result)

if __name__ == "__main__":
    app.run(debug=True)
