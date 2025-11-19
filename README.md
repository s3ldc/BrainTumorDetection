MRI Tumor Detection using Deep Learning

This repository contains an end-to-end deep learning system for automatic MRI brain tumor classification. The project uses a Convolutional Neural Network (EfficientNet-B0) and a Flask-based web application to provide real-time tumor predictions and an interactive evaluation dashboard.

This work was developed as part of a Capstone Project (MCA – CMR University, 2023–2025).

Features

Deep Learning Model (EfficientNet-B0)

Classifies MRI scans into four classes:

glioma

meningioma

notumor

pituitary

Flask Web Application

Upload MRI images and receive predictions instantly.

Displays prediction probability & tumor class.

Interactive Dashboard

Dataset distribution charts

ROC curve

Confusion matrix

Classification report

Training accuracy & loss curves

Included Artifacts

modelfinal.h5 (trained model)

classification_report.json

history.json

test_labels.npy, test_predictions.npy

Repository Structure
CAPSTONE_PROJECT/
├── MRI Images/
│   ├── Training/
│   └── Testing/
├── static/
│   ├── charts/
│   ├── data/
│   ├── uploads/
│   └── style.css
├── templates/
│   ├── index.html
│   └── dashboard.html
├── uploads/
├── Brain-Tumor_Detection.ipynb
├── classification_report.json
├── history.json
├── main.py
├── modelfinal.h5
├── test_labels.npy
├── test_predictions.npy
└── README.md

Tech Stack

Deep Learning:
TensorFlow, Keras, EfficientNet-B0, NumPy, OpenCV, Pillow

Web App:
Flask, HTML5, CSS3, JavaScript, Bootstrap

Visualization:
Matplotlib, Seaborn

Dataset

Publicly available Brain MRI datasets (e.g., Kaggle / BRATS-style).

~3,000–4,000 images

Four tumor-related classes

Preprocessing steps:

Resize to 224×224

Normalize pixel values

Apply augmentation (training only)

Split:

70% Training

20% Validation

10% Testing

Installation & Setup
1. Clone the repository
git clone <your-repo-url>
cd CAPSTONE_PROJECT

2. Create a virtual environment
python -m venv venv


Activate:

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

3. Install dependencies

Create a requirements.txt containing:

flask>=2.0
tensorflow>=2.6
numpy
pillow
opencv-python
matplotlib
scikit-learn
pandas
seaborn
gunicorn


Install:

pip install -r requirements.txt

Running the Application

Start the Flask server:

python main.py


Visit the app at:

http://127.0.0.1:5000/

Available Pages

Homepage: Upload MRI → Get prediction

Dashboard: Evaluation metrics, dataset charts, and model reports

API Usage
Prediction API
POST /predict


Form Data:
file → MRI image file

Sample Response

{
  "prediction": "glioma",
  "confidence": 0.9873
}

Model Performance Overview

AUC: 0.99–1.00 across all classes

Accuracy: High on both training & testing sets

Classification Report:

Precision: 0.93–0.99

Recall: 0.93–1.00

F1 Score: 0.94–0.99

Visual charts are available on the dashboard.

Training Workflow (Notebook)

The notebook (Brain-Tumor_Detection.ipynb) performs:

Dataset loading

Preprocessing & augmentation

EfficientNet-B0 model setup

Training with callbacks

Evaluation (ROC, confusion matrix, reports)

Saving results and exporting artifacts

Future Enhancements

PACS integration (DICOM workflow)

Cloud deployment (AWS/GCP/Azure)

Lightweight mobile version

Explainable AI with Grad-CAM overlays

Tumor segmentation (pixel-level)

Acknowledgements

Guide: Prof. Aurangazeb Khan

Institution: CMR University, School of Science & Computer Studies

Open-source community (TensorFlow, Flask, Bootstrap)

License

This project is part of an academic capstone.
For reuse or research purposes, please credit the author.
