from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

model = joblib.load("student_performance_model.pkl")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        return jsonify({"predicted_score": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
