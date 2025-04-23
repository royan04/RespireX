from flask import Flask, render_template
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
# Load your trained model
model = load_model("best_weight.h5")

# Define the class labels
class_labels = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis', 'Unidentified']  # adjust based on your model

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['image']
    
    try:
        img = Image.open(img_file).convert('RGB')
        img = img.resize((224, 224))  # adjust if your model uses another input size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        return jsonify({'result': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model')
def model_page():
    return render_template('model.html')


@app.route('/result')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
