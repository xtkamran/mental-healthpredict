from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the models
with open("phq_model.pkl", "rb") as f:
    phq_model = pickle.load(f)

with open("gad_model.pkl", "rb") as f:
    gad_model = pickle.load(f)

# Serve the HTML page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    bmi = float(data["bmi"])
    epworth = int(data["epworth_score"])
    suicidal = int(data["suicidal"])
    depressed = int(data["depressiveness"])
    anxious = int(data["anxiousness"])

    input_data = np.array([[bmi, epworth, suicidal, depressed, anxious]])

    phq_score = phq_model.predict(input_data)[0]
    gad_score = gad_model.predict(input_data)[0]

    def phq_severity(score):
        if score <= 4:
            return "None"
        elif score <= 9:
            return "Mild"
        elif score <= 14:
            return "Moderate"
        elif score <= 19:
            return "Moderately Severe"
        else:
            return "Severe"

    def gad_severity(score):
        if score <= 4:
            return "None"
        elif score <= 9:
            return "Mild"
        elif score <= 14:
            return "Moderate"
        else:
            return "Severe"

    return jsonify({
        "predicted_phq_score": round(phq_score, 2),
        "depression_severity": phq_severity(phq_score),
        "predicted_gad_score": round(gad_score, 2),
        "anxiety_severity": gad_severity(gad_score)
    })

if __name__ == '__main__':
    app.run(debug=True)
