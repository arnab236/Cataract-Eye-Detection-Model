from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
import os

from utils.preprocessing import preprocess_image, is_eye_image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("models/severity_model.h5")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Check if image uploaded
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    # Save uploaded image
    upload_path = os.path.join("static/uploads", file.filename)
    file.save(upload_path)

    # Preprocess image
    image, gray = preprocess_image(upload_path)

    # ---------- VALIDATE EYE IMAGE ----------
    if not is_eye_image(image):
        return render_template(
            "index.html",
            result="Not a valid eye image"
        )

    # ---------- PREPARE IMAGE FOR MODEL ----------
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # ---------- MODEL PREDICTION ----------
    prediction = model.predict(img)

    print("Prediction value:", prediction)

    if prediction[0][0] > 0.5:
        result = "Cataract Detected"
    else:
        result = "Normal Eye"

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)