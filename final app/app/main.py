# =========================
# Brain Tumor Detection App
# =========================

import os

# ---- Suppress TensorFlow logs ----
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
import numpy as np
import traceback
import time

from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename


# =========================
# TensorFlow Resource Limits
# =========================

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# =========================
# Flask Setup
# =========================

app = Flask(__name__)

app.secret_key = "brain-tumor-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# =========================
# Model Setup
# =========================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model.h5"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

print("Loading model...")

try:

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")
    print(
        "Model expects input shape:",
        model.input_shape
    )

except Exception as e:

    print("MODEL LOADING ERROR")
    print(str(e))
    traceback.print_exc()
    raise


# =========================
# Class Labels
# =========================

class_labels = [
    "pituitary",
    "glioma",
    "notumor",
    "meningioma"
]

print("Class labels:", class_labels)


# =========================
# Helper Functions
# =========================

def allowed_file(filename):

    if not filename:
        return False

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================
# Prediction Function
# =========================

def predict_tumor(image_path):

    try:

        print("--------------------------------")
        print("Starting prediction...")
        print("Image path:", image_path)

        # -------------------------
        # Get model input size
        # -------------------------

        image_height = int(
            model.input_shape[1]
        )

        image_width = int(
            model.input_shape[2]
        )

        print(
            "Required image size:",
            image_width,
            "x",
            image_height
        )

        # -------------------------
        # Load image
        # -------------------------

        img = load_img(
            image_path,
            target_size=(
                image_height,
                image_width
            ),
            color_mode="rgb"
        )

        print(
            "Image loaded successfully."
        )

        # -------------------------
        # Convert image to array
        # -------------------------

        img_array = img_to_array(img)

        print(
            "Original array shape:",
            img_array.shape
        )

        # -------------------------
        # Normalize
        # -------------------------

        img_array = np.asarray(
            img_array,
            dtype=np.float32
        )

        img_array = img_array / 255.0

        # -------------------------
        # Add batch dimension
        # -------------------------

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        print(
            "Final model input shape:",
            img_array.shape
        )

        # -------------------------
        # Prediction
        # -------------------------

        print(
            "Running model prediction..."
        )

        # Direct inference instead of
        # model.predict() to reduce
        # prediction overhead.

        predictions = model(
            img_array,
            training=False
        )

        # Convert TensorFlow tensor
        # to NumPy array

        predictions = predictions.numpy()

        print(
            "Raw predictions:",
            predictions
        )

        # -------------------------
        # Predicted class
        # -------------------------

        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )

        confidence = float(
            predictions[0][
                predicted_index
            ]
        )

        print(
            "Predicted index:",
            predicted_index
        )

        print(
            "Confidence:",
            confidence
        )

        # -------------------------
        # Safety check
        # -------------------------

        if predicted_index >= len(
            class_labels
        ):

            return (
                "Error: Model returned "
                "an invalid class.",
                0.0
            )

        # -------------------------
        # Get label
        # -------------------------

        label = class_labels[
            predicted_index
        ]

        print(
            "Predicted label:",
            label
        )

        # -------------------------
        # Display result
        # -------------------------

        if label == "notumor":

            result = (
                "No Tumor Detected"
            )

        elif label == "pituitary":

            result = (
                "Tumor: Pituitary"
            )

        elif label == "glioma":

            result = (
                "Tumor: Glioma"
            )

        elif label == "meningioma":

            result = (
                "Tumor: Meningioma"
            )

        else:

            result = (
                f"Tumor: {label}"
            )

        print(
            "Final result:",
            result
        )

        print("--------------------------------")

        return (
            result,
            confidence
        )

    except Exception as e:

        print("--------------------------------")
        print("PREDICTION ERROR")
        print(str(e))

        traceback.print_exc()

        print("--------------------------------")

        return (
            f"Prediction error: {str(e)}",
            0.0
        )


# =========================
# Main Route
# =========================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    print(
        f"Request received: "
        f"{request.method} {request.path}"
    )

    # =========================
    # GET Request
    # =========================

    if request.method == "GET":

        return render_template(
            "index.html",
            result=None,
            confidence=None,
            file_path=None
        )

    # =========================
    # POST Request
    # =========================

    print(
        "POST request received."
    )

    print(
        "Files received:",
        list(request.files.keys())
    )

    # -------------------------
    # Check file field
    # -------------------------

    if "file" not in request.files:

        print(
            "ERROR: 'file' not found "
            "in request."
        )

        return render_template(
            "index.html",
            result="No file uploaded.",
            confidence=None,
            file_path=None
        )

    file = request.files["file"]

    print(
        "Uploaded filename:",
        file.filename
    )

    # -------------------------
    # Check filename
    # -------------------------

    if file.filename == "":

        print(
            "ERROR: Empty filename."
        )

        return render_template(
            "index.html",
            result=(
                "Please select an image."
            ),
            confidence=None,
            file_path=None
        )

    # -------------------------
    # Check extension
    # -------------------------

    if not allowed_file(
        file.filename
    ):

        print(
            "ERROR: Invalid file type:",
            file.filename
        )

        return render_template(
            "index.html",
            result=(
                "Invalid file type. "
                "Please upload PNG, JPG, "
                "or JPEG."
            ),
            confidence=None,
            file_path=None
        )

    # =========================
    # Save Uploaded Image
    # =========================

    filename = secure_filename(
        file.filename
    )

    # Add timestamp to avoid
    # filename conflicts

    filename = (
        str(int(time.time()))
        + "_"
        + filename
    )

    file_location = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        file.save(
            file_location
        )

        print(
            "File saved successfully:"
        )

        print(
            file_location
        )

    except Exception as e:

        print(
            "ERROR saving file:"
        )

        traceback.print_exc()

        return render_template(
            "index.html",
            result=(
                f"Error saving image: "
                f"{str(e)}"
            ),
            confidence=None,
            file_path=None
        )

    # =========================
    # Prediction
    # =========================

    result, confidence = (
        predict_tumor(
            file_location
        )
    )

    # =========================
    # Return Result
    # =========================

    return render_template(
        "index.html",
        result=result,
        confidence=(
            f"{confidence * 100:.2f}%"
        ),
        file_path=(
            f"/uploads/{filename}"
        )
    )


# =========================
# Uploaded Images
# =========================

@app.route(
    "/uploads/<filename>"
)
def get_uploaded_file(
    filename
):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# =========================
# Health Check
# =========================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": (
            model is not None
        ),
        "model_input_shape": str(
            model.input_shape
        )
    }


# =========================
# Run Locally
# =========================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
