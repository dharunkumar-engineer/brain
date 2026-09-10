TEAM-43 
🧠 Brain Tumor Detection using Deep Learning (VGG16 + Flask)

This project is a web-based Brain Tumor Detection System that uses a pre-trained VGG16 deep learning model to classify MRI brain images.
The application allows users to upload an MRI scan and instantly view whether a tumor is present—making it helpful for academic, research, and demonstration purposes.

📌 Features

✔ Upload MRI images (JPG, PNG)
✔ Pre-processing: resizing, normalization
✔ VGG16-based deep learning prediction
✔ Displays:

Tumor / No Tumor

Type of tumor (Glioma / Meningioma / Pituitary / No Tumor)

Confidence percentage
✔ Simple and user-friendly Flask interface


🛠 Project Structure
Brain-Tumor-Detection/
│
├── models/
│   └── model.h5
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── app.py
├── requirements.txt
└── README.md

⚙️ Requirements

Install Python (version 3.8+ recommended)

Required libraries (install via requirements.txt):

Flask
tensorflow
keras
numpy
pillow
opencv-python
werkzeug


To install all dependencies:
pip install -r requirements.txt

▶️ How to Run the Project (Step-by-Step)
1️⃣ Download the project files

Place app.py, model.h5, and the templates/ folder together.
Ensure the folder structure matches what is shown above.

2️⃣ Install Dependencies

Open a terminal or command prompt inside the project folder:
pip install -r requirements.txt
If you don’t have a requirements file, run:
pip install flask tensorflow keras pillow numpy opencv-python werkzeug

3️⃣ Run the Flask Application

Run the following command:
python app.py
You should see:
Running on http://127.0.0.1:5000/

4️⃣ Open the Web App

Open your browser and go to:
http://127.0.0.1:5000/

5️⃣ Upload an MRI Image
Upload a brain MRI scan
Click Predict
View results instantly (Tumor type + confidence score)

🧪 Model Details

Architecture: VGG16 (pretrained on ImageNet)
Layers Fine-Tuned: last 2–3 convolution layers
Image Size: 128 × 128
Output Classes:
Glioma
Meningioma
Pituitary
No Tumor

👨‍💻 Author

Prepared by DHARUNKUMAR C
Department of Information Technology
College PIET-Parul University
