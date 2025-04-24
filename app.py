from flask import Flask, render_template, request, jsonify, send_file, url_for, redirect 

from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from datetime import datetime
import io
from reportlab.pdfgen import canvas
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Frame
from reportlab.lib.enums import TA_CENTER
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash


app = Flask(__name__)

# Config for database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Load the trained model
model = load_model("best_weight.h5")

# Define class labels
class_labels = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']

# Define the database model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120))
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow) 

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'doctor' or 'patient'

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['image']
    filename = img_file.filename

    try:
        img = Image.open(img_file).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        # Save to database
        new_prediction = Prediction(filename=filename, prediction=predicted_class, confidence=confidence)
        db.session.add(new_prediction)
        db.session.commit()

        return jsonify({'result': predicted_class, 'confidence': f"{confidence:.2f}"})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/download/<int:prediction_id>')
def download_report(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)

    buffer = BytesIO()
    generate_medical_report(
        file_name=prediction.filename,
        prediction=prediction.prediction,
        confidence=prediction.confidence,
        save_path=buffer  # Write to BytesIO instead of a physical file
    )
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{prediction.filename.split('.')[0]}_report.pdf",
        mimetype='application/pdf'
    )

def generate_medical_report(file_name, prediction, confidence, save_path):
    from reportlab.lib.colors import HexColor

    # Dummy patient info
    patient_name = "NA"
    age = "NA"
    gender = "NA"
    report_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Disease descriptions dictionary
    disease_info = {
        "Pneumonia": {
            "overview": "Pneumonia is an infection that inflames the air sacs in one or both lungs. "
                        "It may cause cough with phlegm or pus, fever, chills, and difficulty breathing.",
            "cause": "Caused by bacteria, viruses, or fungi. Risk increases in children, elderly, or people with weakened immunity.",
            "advice": "Seek immediate medical consultation. Treatment may involve antibiotics or antiviral medications."
        },
        "Tuberculosis": {
            "overview": "Tuberculosis (TB) is a potentially serious infectious disease that mainly affects the lungs. "
                        "The bacteria that cause TB are spread through tiny droplets released into the air.",
            "cause": "Caused by Mycobacterium tuberculosis. Risk increases in overcrowded or poorly ventilated areas.",
            "advice": "Consult a pulmonologist. TB is treatable with a strict antibiotic regimen over several months."
        },
        "COVID-19": {
            "overview": "COVID-19 is a contagious disease caused by the SARS-CoV-2 virus, affecting primarily the respiratory system.",
            "cause": "Spread through respiratory droplets. Causes symptoms like fever, cough, and shortness of breath.",
            "advice": "Immediate isolation and consultation with a physician is advised. Antiviral treatment may be required."
        }
    }

    disease_data = disease_info.get(prediction, {
        "overview": "No information available for this disease.",
        "cause": "",
        "advice": "Consult a healthcare provider for more information."
    })

    # Determine risk level
    if confidence > 90:
        risk = "🟥 High"
    elif confidence > 70:
        risk = "🟧 Moderate"
    else:
        risk = "🟩 Low"

    # Set up canvas
    c = canvas.Canvas(save_path, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 60, "🫁 RespireX Medical Report")

    # Draw box for patient info
    c.setStrokeColor(HexColor("#999999"))
    c.rect(inch, height - 130, width - 2 * inch, 60, stroke=1, fill=0)
    c.setFont("Helvetica", 12)
    c.drawString(inch + 10, height - 100, f"Patient Name: {patient_name}")
    c.drawString(inch + 220, height - 100, f"Age: {age}")
    c.drawString(inch + 320, height - 100, f"Gender: {gender}")
    c.drawString(inch + 10, height - 115, f"Report ID: {report_id}")
    c.drawString(inch + 250, height - 115, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Draw horizontal line
    c.line(inch, height - 140, width - inch, height - 140)

    # Prediction & Confidence
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, height - 170, f"🩺 Prediction Summary")
    c.setFont("Helvetica", 12)
    c.drawString(inch + 20, height - 190, f"Diagnosis: {prediction}")
    c.drawString(inch + 20, height - 210, f"Confidence: {confidence:.2f}%")
    c.drawString(inch + 20, height - 230, f"Risk Level: {risk}")

    # Draw horizontal line
    c.line(inch, height - 245, width - inch, height - 245)

    # Disease Details Box
    text_y = height - 270
    text_width = width - 2 * inch
    styles = getSampleStyleSheet()
    body_style = styles["BodyText"]
    body_style.fontSize = 11
    body_style.leading = 16

    disease_text = f"""
    <b>Overview:</b><br/>{disease_data['overview']}<br/><br/>
    <b>Cause & Risk:</b><br/>{disease_data['cause']}<br/><br/>
    <b>Medical Advice:</b><br/>{disease_data['advice']}
    """
    p = Paragraph(disease_text, body_style)
    f = Frame(inch, inch + 80, text_width, text_y - inch - 80, showBoundary=0)
    f.addFromList([p], c)

    # Footer & Disclaimer
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(width / 2, 60, "Disclaimer: This is an AI-generated report. Please consult a licensed medical professional.")
    c.drawCentredString(width / 2, 45, "© 2025 RespireX AI Diagnostic System v1.0")

    c.showPage()
    c.save()

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup')
def signup():
    return render_template('login.html')

# Route for model page
@app.route('/model')
def model_page():
    return render_template('doctors.html')

@app.route('/signout')
def sign_out():
    return render_template('index.html')

# Route for result page
@app.route('/result')
def result():
    return render_template('result.html')

# Route for report page
@app.route('/report')
def report():
    all_predictions = Prediction.query.order_by(Prediction.id.desc()).all()
    return render_template('report.html', predictions=all_predictions)

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    role = request.form['role']

    user = User.query.filter_by(email=email, role=role).first()

    if user and check_password_hash(user.password, password):
        if role == 'doctor':
            return redirect('/model')
        elif role == 'patient':
            return redirect('/report')
    return "Invalid credentials or role mismatch", 401

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Ensure tables are created before the app starts
    app.run(debug=True)