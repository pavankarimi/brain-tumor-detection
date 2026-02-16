import os
import numpy as np
import gdown
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"

# Create uploads folder if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# ===============================
# 🔥 Download model from Drive
# ===============================

MODEL_PATH = "brain_tumor_vgg16.h5"
FILE_ID = "1Ynr8khfe7OcbAg9U86swblYGpen9PfTr"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print("Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)

# Load trained model
model = load_model(MODEL_PATH)

IMG_SIZE = 224


def predict_tumor(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    prediction = float(prediction)

    if prediction > 0.5:
        result = "Tumor Detected"
        confidence = prediction * 100
    else:
        result = "No Tumor"
        confidence = (1 - prediction) * 100

    return result, round(confidence, 2)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    name = request.form.get("name")
    age = request.form.get("age")
    file = request.files.get("image")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result, confidence = predict_tumor(filepath)

    return jsonify({
        "name": name,
        "age": age,
        "result": result,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)
