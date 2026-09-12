# =========================
# Brain Tumor Detection App
# =========================

import os

# ---- Suppress TensorFlow logs ----
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import sqlite3
import traceback
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    send_file,
    redirect,
    url_for,
    flash
)

from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage
)

from PIL import Image


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

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "brain-tumor-secret-key"
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)


REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "reports"
)


DATABASE_PATH = os.path.join(
    BASE_DIR,
    "brain_tumor.db"
)


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["REPORT_FOLDER"] = REPORT_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}


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

    print(
        "Model loaded successfully."
    )

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


print(
    "Class labels:",
    class_labels
)


# =========================
# Model Accuracy
# =========================

# Fixed model evaluation accuracy.
# This is the only model metric displayed
# on the dashboard.

MODEL_ACCURACY = "96.9"


# =========================
# Database
# =========================

def get_db_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_db_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            patient_name TEXT NOT NULL,

            patient_id TEXT NOT NULL,

            age INTEGER NOT NULL,

            gender TEXT NOT NULL,

            image_path TEXT NOT NULL,

            result TEXT NOT NULL,

            confidence TEXT NOT NULL,

            date TEXT NOT NULL

        )
        """
    )

    connection.commit()

    connection.close()

    print(
        "SQLite database initialized."
    )


init_database()


# =========================
# Helper Functions
# =========================

def allowed_file(filename):

    if not filename:

        return False

    return (
        "."
        in filename
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

        print(
            "Starting prediction..."
        )

        print(
            "Image path:",
            image_path
        )


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

        img_array = img_to_array(
            img
        )


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


        img_array = (
            img_array / 255.0
        )


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


        predictions = model(
            img_array,
            training=False
        )


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

        print(
            "PREDICTION ERROR"
        )

        print(
            str(e)
        )

        traceback.print_exc()

        print("--------------------------------")


        return (
            f"Prediction error: {str(e)}",
            0.0
        )


# =========================
# Save Scan
# =========================

def save_scan(
    patient_name,
    patient_id,
    age,
    gender,
    image_path,
    result,
    confidence
):

    connection = get_db_connection()

    cursor = connection.cursor()


    scan_date = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )


    cursor.execute(
        """
        INSERT INTO scans (
            patient_name,
            patient_id,
            age,
            gender,
            image_path,
            result,
            confidence,
            date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_name,
            patient_id,
            age,
            gender,
            image_path,
            result,
            f"{confidence * 100:.2f}%",
            scan_date
        )
    )


    scan_id = cursor.lastrowid


    connection.commit()

    connection.close()


    return scan_id


# =========================
# Get Scan
# =========================

def get_scan(scan_id):

    connection = get_db_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *
        FROM scans
        WHERE id = ?
        """,
        (scan_id,)
    )


    scan = cursor.fetchone()


    connection.close()


    return scan


# =========================
# Get Scans
# =========================

def get_scans(search=""):

    connection = get_db_connection()

    cursor = connection.cursor()


    if search:

        search_value = (
            f"%{search}%"
        )


        cursor.execute(
            """
            SELECT *
            FROM scans
            WHERE patient_name LIKE ?
               OR patient_id LIKE ?
            ORDER BY id DESC
            """,
            (
                search_value,
                search_value
            )
        )


    else:

        cursor.execute(
            """
            SELECT *
            FROM scans
            ORDER BY id DESC
            """
        )


    scans = cursor.fetchall()


    connection.close()


    return scans


# =========================
# Dashboard Statistics
# =========================

def get_dashboard_statistics():

    connection = get_db_connection()

    cursor = connection.cursor()


    # -------------------------
    # Total scans
    # -------------------------

    cursor.execute(
        "SELECT COUNT(*) FROM scans"
    )

    total_scans = (
        cursor.fetchone()[0]
    )


    # -------------------------
    # Tumor cases
    # -------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM scans
        WHERE result LIKE 'Tumor:%'
        """
    )

    tumor_cases = (
        cursor.fetchone()[0]
    )


    # -------------------------
    # No tumor cases
    # -------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM scans
        WHERE result = 'No Tumor Detected'
        """
    )

    no_tumor_cases = (
        cursor.fetchone()[0]
    )


    # -------------------------
    # Glioma
    # -------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM scans
        WHERE result = 'Tumor: Glioma'
        """
    )

    glioma_cases = (
        cursor.fetchone()[0]
    )


    # -------------------------
    # Meningioma
    # -------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM scans
        WHERE result = 'Tumor: Meningioma'
        """
    )

    meningioma_cases = (
        cursor.fetchone()[0]
    )


    # -------------------------
    # Pituitary
    # -------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM scans
        WHERE result = 'Tumor: Pituitary'
        """
    )

    pituitary_cases = (
        cursor.fetchone()[0]
    )


    connection.close()


    return {

        "total_scans":
            total_scans,

        "tumor_cases":
            tumor_cases,

        "no_tumor_cases":
            no_tumor_cases,

        "glioma_cases":
            glioma_cases,

        "meningioma_cases":
            meningioma_cases,

        "pituitary_cases":
            pituitary_cases,

        "accuracy":
            MODEL_ACCURACY
    }


# =========================
# Main Route
# =========================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    # =========================
    # POST
    # =========================

    if request.method == "POST":

        patient_name = request.form.get(
            "patient_name",
            ""
        ).strip()


        patient_id = request.form.get(
            "patient_id",
            ""
        ).strip()


        age = request.form.get(
            "age",
            ""
        ).strip()


        gender = request.form.get(
            "gender",
            ""
        ).strip()


        # -------------------------
        # Validation
        # -------------------------

        if not patient_name:

            flash(
                "Please enter patient name."
            )

            return redirect(
                url_for("index")
            )


        if not patient_id:

            flash(
                "Please enter patient ID."
            )

            return redirect(
                url_for("index")
            )


        if not age:

            flash(
                "Please enter patient age."
            )

            return redirect(
                url_for("index")
            )


        if not gender:

            flash(
                "Please select gender."
            )

            return redirect(
                url_for("index")
            )


        try:

            age = int(age)


            if age < 1 or age > 120:

                raise ValueError


        except ValueError:

            flash(
                "Please enter a valid age."
            )

            return redirect(
                url_for("index")
            )


        # -------------------------
        # File validation
        # -------------------------

        if "file" not in request.files:

            flash(
                "No MRI image uploaded."
            )

            return redirect(
                url_for("index")
            )


        file = request.files[
            "file"
        ]


        if file.filename == "":

            flash(
                "Please select an MRI image."
            )

            return redirect(
                url_for("index")
            )


        if not allowed_file(
            file.filename
        ):

            flash(
                "Invalid file type. "
                "Please upload PNG, JPG, "
                "or JPEG."
            )

            return redirect(
                url_for("index")
            )


        # =========================
        # Save Image
        # =========================

        original_filename = (
            secure_filename(
                file.filename
            )
        )


        filename = (
            str(
                int(
                    time.time() * 1000
                )
            )
            + "_"
            + original_filename
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
                "Uploaded image saved:",
                file_location
            )


            print(
                "Image exists:",
                os.path.exists(
                    file_location
                )
            )


        except Exception as e:

            traceback.print_exc()


            flash(
                f"Error saving image: {str(e)}"
            )


            return redirect(
                url_for("index")
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
        # Save SQLite Record
        # =========================

        # IMPORTANT:
        # Store ONLY the filename.
        #
        # Correct:
        # 1757654321000_mri.jpg
        #
        # Incorrect:
        # /uploads/1757654321000_mri.jpg
        #
        # Incorrect:
        # C:/project/uploads/1757654321000_mri.jpg

        scan_id = save_scan(
            patient_name,
            patient_id,
            age,
            gender,
            filename,
            result,
            confidence
        )


        return redirect(
            url_for(
                "index",
                scan_id=scan_id
            )
        )


    # =========================
    # GET
    # =========================

    scan_id = request.args.get(
        "scan_id",
        type=int
    )


    search = request.args.get(
        "search",
        ""
    ).strip()


    result = None

    confidence = None

    file_path = None

    patient_name = None

    patient_id = None

    age = None

    gender = None


    # =========================
    # Latest Scan
    # =========================

    if scan_id:

        scan = get_scan(
            scan_id
        )


        if scan:

            result = scan[
                "result"
            ]


            confidence = scan[
                "confidence"
            ]


            patient_name = scan[
                "patient_name"
            ]


            patient_id = scan[
                "patient_id"
            ]


            age = scan[
                "age"
            ]


            gender = scan[
                "gender"
            ]


            # IMPORTANT:
            # Keep this as ONLY the
            # filename.
            #
            # The HTML handles:
            # url_for(
            #   'get_uploaded_file',
            #   filename=file_path
            # )

            file_path = scan[
                "image_path"
            ]


    # =========================
    # Dashboard
    # =========================

    stats = (
        get_dashboard_statistics()
    )


    # =========================
    # Previous Scans
    # =========================

    scans = get_scans(
        search
    )


    return render_template(

        "index.html",

        result=result,

        confidence=confidence,

        file_path=file_path,

        patient_name=patient_name,

        patient_id=patient_id,

        age=age,

        gender=gender,

        scan_id=scan_id,

        scans=scans,

        search=search,

        total_scans=stats[
            "total_scans"
        ],

        tumor_cases=stats[
            "tumor_cases"
        ],

        no_tumor_cases=stats[
            "no_tumor_cases"
        ],

        glioma_cases=stats[
            "glioma_cases"
        ],

        meningioma_cases=stats[
            "meningioma_cases"
        ],

        pituitary_cases=stats[
            "pituitary_cases"
        ],

        accuracy=stats[
            "accuracy"
        ]

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

        app.config[
            "UPLOAD_FOLDER"
        ],

        filename

    )


# =========================
# View Scan
# =========================

@app.route(
    "/view_scan/<int:scan_id>"
)
def view_scan(scan_id):

    scan = get_scan(
        scan_id
    )


    if not scan:

        flash(
            "Scan record not found."
        )

        return redirect(
            url_for("index")
        )


    return render_template(
        "scan.html",
        scan=scan
    )


# =========================
# Delete Scan
# =========================

@app.route(
    "/delete_scan/<int:scan_id>",
    methods=["POST"]
)
def delete_scan(scan_id):

    scan = get_scan(
        scan_id
    )


    if not scan:

        flash(
            "Scan record not found."
        )

        return redirect(
            url_for("index")
        )


    # -------------------------
    # Delete image
    # -------------------------

    image_path = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        scan[
            "image_path"
        ]

    )


    if os.path.exists(
        image_path
    ):

        try:

            os.remove(
                image_path
            )

        except Exception:

            pass


    # -------------------------
    # Delete database record
    # -------------------------

    connection = (
        get_db_connection()
    )


    cursor = (
        connection.cursor()
    )


    cursor.execute(

        """
        DELETE FROM scans
        WHERE id = ?
        """,

        (scan_id,)

    )


    connection.commit()

    connection.close()


    flash(
        "Scan deleted successfully."
    )


    return redirect(
        url_for("index")
    )


# =========================
# PDF Report
# =========================

@app.route(
    "/download_report/<int:scan_id>"
)
def download_report(scan_id):

    scan = get_scan(
        scan_id
    )


    if not scan:

        flash(
            "Scan record not found."
        )

        return redirect(
            url_for("index")
        )


    safe_patient_name = (
        secure_filename(
            scan["patient_name"]
        )
    )


    if not safe_patient_name:

        safe_patient_name = "Patient"


    pdf_filename = (

        "Brain_Tumor_Report_"

        + safe_patient_name

        + "_"

        + str(scan_id)

        + ".pdf"

    )


    pdf_path = os.path.join(

        app.config[
            "REPORT_FOLDER"
        ],

        pdf_filename

    )


    try:

        document = SimpleDocTemplate(

            pdf_path,

            pagesize=A4,

            rightMargin=40,

            leftMargin=40,

            topMargin=40,

            bottomMargin=40

        )


        styles = (
            getSampleStyleSheet()
        )


        title_style = ParagraphStyle(

            "TitleStyle",

            parent=styles["Title"],

            alignment=TA_CENTER,

            fontSize=20,

            spaceAfter=15

        )


        heading_style = ParagraphStyle(

            "HeadingStyle",

            parent=styles["Heading2"],

            fontSize=14,

            spaceBefore=10,

            spaceAfter=10

        )


        normal_style = ParagraphStyle(

            "NormalStyle",

            parent=styles["Normal"],

            fontSize=10,

            leading=15

        )


        story = []


        # -------------------------
        # Title
        # -------------------------

        story.append(

            Paragraph(

                "BRAIN TUMOR DETECTION REPORT",

                title_style

            )

        )


        story.append(

            Paragraph(

                "AI-Based MRI Analysis System",

                ParagraphStyle(

                    "Subtitle",

                    parent=normal_style,

                    alignment=TA_CENTER,

                    fontSize=11

                )

            )

        )


        story.append(
            Spacer(1, 20)
        )


        # -------------------------
        # Report ID
        # -------------------------

        story.append(

            Paragraph(

                f"<b>Report ID:</b> "
                f"BT-{scan_id:05d}",

                normal_style

            )

        )


        story.append(
            Spacer(1, 10)
        )


        # -------------------------
        # Patient Details
        # -------------------------

        story.append(

            Paragraph(

                "Patient Details",

                heading_style

            )

        )


        patient_data = [

            [
                "Patient Name",
                str(scan["patient_name"])
            ],

            [
                "Patient ID",
                str(scan["patient_id"])
            ],

            [
                "Age",
                str(scan["age"])
            ],

            [
                "Gender",
                str(scan["gender"])
            ],

            [
                "Scan Date",
                str(scan["date"])
            ]

        ]


        patient_table = Table(

            patient_data,

            colWidths=[
                150,
                330
            ]

        )


        patient_table.setStyle(

            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )

            ])

        )


        story.append(
            patient_table
        )


        story.append(
            Spacer(1, 20)
        )


        # -------------------------
        # MRI Image
        # -------------------------

        story.append(

            Paragraph(
                "MRI Image",
                heading_style
            )

        )


        image_path = os.path.join(

            app.config[
                "UPLOAD_FOLDER"
            ],

            scan[
                "image_path"
            ]

        )


        if os.path.exists(
            image_path
        ):

            try:

                with Image.open(
                    image_path
                ) as pil_image:

                    width, height = (
                        pil_image.size
                    )


                if width > 0 and height > 0:

                    max_width = (
                        4.5 * inch
                    )

                    max_height = (
                        4.5 * inch
                    )


                    scale = min(

                        max_width / width,

                        max_height / height

                    )


                    report_width = (
                        width * scale
                    )


                    report_height = (
                        height * scale
                    )


                    mri_image = (
                        ReportLabImage(

                            image_path,

                            width=report_width,

                            height=report_height

                        )
                    )


                    story.append(
                        mri_image
                    )


                    story.append(
                        Spacer(1, 15)
                    )


            except Exception as e:

                print(
                    "Could not add MRI image:",
                    e
                )


        # -------------------------
        # Prediction
        # -------------------------

        story.append(

            Paragraph(

                "Prediction Result",

                heading_style

            )

        )


        prediction_data = [

            [
                "Prediction",
                str(scan["result"])
            ],

            [
                "Confidence Score",
                str(scan["confidence"])
            ]

        ]


        prediction_table = Table(

            prediction_data,

            colWidths=[
                150,
                330
            ]

        )


        prediction_table.setStyle(

            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )

            ])

        )


        story.append(
            prediction_table
        )


        story.append(
            Spacer(1, 25)
        )


        # -------------------------
        # System Information
        # -------------------------

        story.append(

            Paragraph(

                "System Information",

                heading_style

            )

        )


        system_data = [

            [
                "Model",
                "VGG16"
            ],

            [
                "Framework",
                "TensorFlow / Keras"
            ],

            [
                "Model Accuracy",
                "96.9%"
            ],

            [
                "Classes",
                "Pituitary, Glioma, "
                "No Tumor, Meningioma"
            ]

        ]


        system_table = Table(

            system_data,

            colWidths=[
                150,
                330
            ]

        )


        system_table.setStyle(

            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )

            ])

        )


        story.append(
            system_table
        )


        story.append(
            Spacer(1, 25)
        )


        # -------------------------
        # Disclaimer
        # -------------------------

        story.append(

            Paragraph(

                "<b>Disclaimer:</b> This report is "
                "generated by an AI-based image "
                "classification system and is intended "
                "for educational and research purposes. "
                "It should not be used as a substitute "
                "for professional medical diagnosis.",

                normal_style

            )

        )


        story.append(
            Spacer(1, 15)
        )


        story.append(

            Paragraph(

                "Brain Tumor Detection System - "
                "Team IT 43, Parul University",

                ParagraphStyle(

                    "Footer",

                    parent=normal_style,

                    alignment=TA_CENTER

                )

            )

        )


        document.build(
            story
        )


    except Exception as e:

        print(
            "PDF GENERATION ERROR"
        )

        traceback.print_exc()


        flash(
            f"Could not generate PDF: {str(e)}"
        )


        return redirect(
            url_for("index")
        )


    return send_file(

        pdf_path,

        as_attachment=True,

        download_name=pdf_filename,

        mimetype="application/pdf"

    )


# =========================
# Health Check
# =========================

@app.route(
    "/health"
)
def health():

    return {

        "status":
            "ok",

        "model_loaded":
            model is not None,

        "model_input_shape":
            str(model.input_shape),

        "model_accuracy":
            "96.9%",

        "database":
            os.path.exists(
                DATABASE_PATH
            )

    }


# =========================
# Run
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
