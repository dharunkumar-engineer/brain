# =========================
# Brain Tumor Detection App
# =========================

# ---- Suppress TensorFlow logs ----
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disable oneDNN logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # suppress TF info/warnings

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # suppress absl warnings
# ----------------------------------

from flask import Flask, render_template, request, send_from_directory, flash
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename
import traceback

# =========================
# Flask Setup
# =========================
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Uploads folder setup
UPLOAD_FOLDER = os.path.abspath("./uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# =========================
# Model Setup
# =========================
MODEL_PATH = os.path.abspath("models/model.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load the model only for inference (no compile)
model = load_model(MODEL_PATH, compile=False)

print("Model loaded successfully.")
print("Model expects input shape:", model.input_shape)

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']


# =========================
# Helper Functions
# =========================
def allowed_file(filename):
    """Check if the uploaded file is valid."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_tumor(image_path):
    """Load an image, preprocess it, and make prediction."""
    try:
        # Match the model's expected input size
        IMAGE_SIZE = model.input_shape[1]
        image_path = os.path.abspath(image_path)

        # Preprocess the image
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Predict
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = float(np.max(predictions))

        # Get label
        label = class_labels[predicted_class_index]
        return ("No Tumor Detected", confidence_score) if label == 'notumor' else (f"Tumor: {label}", confidence_score)

    except Exception as e:
        print("❌ Prediction error:\n", traceback.format_exc())
        return f"Error in prediction: {str(e)}", 0.0


# =========================
# Routes
# =========================
@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route for uploading and predicting images."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part")
            return render_template('index.html', result=None)

        file = request.files['file']
        if file.filename == '':
            flash("No selected file")
            return render_template('index.html', result=None)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            try:
                file.save(file_location)
            except Exception as e:
                flash(f"Error saving file: {e}")
                return render_template('index.html', result=None)

            # Make prediction
            result, confidence = predict_tumor(file_location)

            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence * 100:.2f}%",
                file_path=f'/uploads/{filename}'
            )
        else:
            flash("Invalid file type. Allowed types: png, jpg, jpeg")

    return render_template('index.html', result=None)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# =========================
# Run App
# =========================
if __name__ == '__main__':
    # Development server (shows warning but OK for testing)
    app.run(debug=True, host="0.0.0.0", port=5000)
