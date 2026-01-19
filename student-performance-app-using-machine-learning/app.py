from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
from fpdf import FPDF  # Added for PDF export
import os
from werkzeug.utils import secure_filename
from ocr_utils import extract_scores_from_image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure uploads folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

CORS(app)

# Load trained model
model = joblib.load("student_performance_model.pkl")

columns = [
    "hours_studied", "attendance", "parental_involvement", "access_to_resources",
    "extracurricular_activities", "sleep_hours", "previous_scores", "motivation_level",
    "internet_access", "tutoring_sessions", "family_income", "teacher_quality",
    "school_type", "peer_influence", "physical_activity", "learning_disabilities",
    "parental_education_level", "distance_from_home", "gender"
]

# Mappings
parental_map = {"low": 1, "medium": 3, "high": 5}
resource_map = {"low": 1, "medium": 3, "high": 5}
yes_no_map = {"yes": 1, "no": 0}
motivation_map = {"low": 1, "medium": 3, "high": 5}
school_map = {"public": 0, "private": 1}
peer_map = {"negative": 0, "neutral": 2, "positive": 5}
learning_map = {"yes": 0, "no": 1}
edu_map = {"high school": 1, "college": 3, "postgraduate": 5}
gender_map = {"male": 1, "female": 2, "other": 3}
distance_map = {"far": 1, "moderate": 3, "near": 5}
income_map = {"low": 1, "medium": 3, "high": 5}
teacher_map = {"low": 1, "medium": 3, "high": 5}

# ---------------- GPA CONVERSION (NEW ADDITION) ----------------
def percentage_to_gpa(percentage):
    if percentage >= 85:
        return 4.00
    elif percentage >= 80:
        return 3.70
    elif percentage >= 75:
        return 3.30
    elif percentage >= 70:
        return 3.00
    elif percentage >= 65:
        return 2.70
    elif percentage >= 60:
        return 2.30
    else:
        return 2.00
# ---------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Safely get data with defaults if missing
        def get_val(key, default=0):
            return data.get(key, default)

        # Use OCR-extracted scores if provided
        previous_score = float(get_val("Previous_Scores", 50))
        previous_cgpa = float(get_val("Previous_CGPA", 3.0))

        features = [
            float(get_val("Hours_Studied", 0)),
            float(get_val("Attendance", 0)),
            parental_map.get(get_val("Parental_Involvement", "medium").lower(), 3),
            resource_map.get(get_val("Access_to_Resources", "medium").lower(), 3),
            yes_no_map.get(get_val("Extracurricular_Activities", "no").lower(), 0),
            float(get_val("Sleep_Hours", 7)),
            previous_score,  # Use OCR value
            motivation_map.get(get_val("Motivation_Level", "medium").lower(), 3),
            yes_no_map.get(get_val("Internet_Access", "yes").lower(), 1),
            int(get_val("Tutoring_Sessions", 0)),
            income_map.get(get_val("Family_Income", "medium").lower(), 3),
            teacher_map.get(get_val("Teacher_Quality", "medium").lower(), 3),
            school_map.get(get_val("School_Type", "public").lower(), 0),
            peer_map.get(get_val("Peer_Influence", "neutral").lower(), 2),
            int(get_val("Physical_Activity", 0)),
            learning_map.get(get_val("Learning_Disabilities", "no").lower(), 1),
            edu_map.get(get_val("Parental_Education_Level", "college").lower(), 3),
            distance_map.get(get_val("Distance_from_Home", "moderate").lower(), 3),
            gender_map.get(get_val("Gender", "male").lower(), 1)
        ]

        features_df = pd.DataFrame([features], columns=columns)

        # Model prediction
        raw_prediction = model.predict(features_df)[0]

        # Ensure predicted score is not less than extracted OCR score
        score = max(raw_prediction, previous_score)

        # Post-processing boost
        if previous_score >= 85:
            score += 12
        if float(get_val("Hours_Studied", 0)) >= 8:
            score += 10
        if float(get_val("Attendance", 0)) >= 90:
            score += 8
        if get_val("Motivation_Level", "medium").lower() == "high":
            score += 5

        score = round(min(100, max(35, score)), 2)

        # ---------------- GPA / SGPA / CGPA (NEW ADDITION) ----------------
        predicted_gpa = percentage_to_gpa(score)
        predicted_sgpa = predicted_gpa
        predicted_cgpa = round((previous_cgpa + predicted_sgpa) / 2, 2)
        # -----------------------------------------------------------------

        # Trend analysis
        previous = previous_score
        current = score
        if current > previous:
            trend = "up"
            message = "Excellent work! 📈 Your performance is improving. Keep it up!"
        elif current < previous:
            trend = "down"
            message = "Don’t be discouraged 💪. Focus more and you can improve your score."
        else:
            trend = "same"
            message = "Your performance is stable 👍. Push a little more for improvement."

        # Graph generation
        plt.figure(figsize=(5, 3))
        plt.plot(["Previous Score", "Predicted Score"], [previous, current], marker="o")
        plt.ylim(0, 100)
        plt.title("Student Performance Trend")
        plt.ylabel("Score")
        plt.grid(True)

        img = io.BytesIO()
        plt.savefig(img, format="png", bbox_inches="tight")
        img.seek(0)
        graph = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return jsonify({
            "predicted_score": current,
            "predicted_gpa": predicted_gpa,
            "predicted_sgpa": predicted_sgpa,
            "predicted_cgpa": predicted_cgpa,
            "trend": trend,
            "message": message,
            "graph": graph
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)})

# ------------------- PDF/Report Route -------------------
@app.route("/download_report", methods=["POST"])
def download_report():
    try:
        data = request.get_json()
        predicted_score = data.get("predicted_score", 0)
        predicted_gpa = data.get("predicted_gpa", 0)
        predicted_sgpa = data.get("predicted_sgpa", 0)
        predicted_cgpa = data.get("predicted_cgpa", 0)
        message = data.get("message", "")
        graph_base64 = data.get("graph", "")

        graph_bytes = base64.b64decode(graph_base64)

        with open("temp_graph.png", "wb") as f:
            f.write(graph_bytes)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Student Performance Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, f"Predicted Percentage: {predicted_score}%")
        pdf.multi_cell(0, 8, f"Predicted GPA: {predicted_gpa}")
        pdf.multi_cell(0, 8, f"Predicted SGPA: {predicted_sgpa}")
        pdf.multi_cell(0, 8, f"Predicted CGPA: {predicted_cgpa}")
        pdf.multi_cell(0, 8, f"Message: {message}")
        pdf.ln(10)

        pdf.image("temp_graph.png", x=30, w=150)
        pdf.output("student_report.pdf")

        return jsonify({"success": True, "message": "Report generated as student_report.pdf"})

    except Exception as e:
        print("PDF Error:", e)
        return jsonify({"error": str(e)})

# ------------------- OCR Upload Route -------------------
@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    filename = secure_filename(file.filename)

    # Create folder if it does not exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # Call OCR utility
    extracted_data = extract_scores_from_image(path)

    return jsonify({
        "message": "Image processed successfully",
        "extracted_data": extracted_data
    })

# --------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
