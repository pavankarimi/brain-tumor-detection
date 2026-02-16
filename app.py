import os
import numpy as np
import gdown
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==========================================
# App Configuration
# ==========================================

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "brain_tumor_vgg16.h5"
FILE_ID = "1Ynr8khfe7OcbAg9U86swblYGpen9PfTr"
IMG_SIZE = 224

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ==========================================
# Download Model (Only If Not Exists)
# ==========================================

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# ==========================================
# Load Model Safely
# ==========================================

def load_trained_model():
    try:
        download_model()
        print("Loading model...")
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully ✅")
        return model
    except Exception as e:
        print("Model loading failed:", str(e))
        raise e

model = load_trained_model()

# ==========================================
# Prediction Function
# ==========================================

def predict_tumor(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = float(model.predict(img_array)[0][0])

    if prediction > 0.5:
        result = "Tumor Detected"
        confidence = prediction * 100
    else:
        result = "No Tumor"
        confidence = (1 - prediction) * 100

    return result, round(confidence, 2)

# ==========================================
# Routes
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        name = request.form.get("name")
        age = request.form.get("age")
        file = request.files.get("image")

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        result, confidence = predict_tumor(filepath)

        return jsonify({
            "name": name,
            "age": age,
            "result": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# Run App (For Local Only)
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
