MRI Tumor Detection using Deep Learning

This repository contains an end-to-end deep learning system for automatic MRI brain tumor classification. The project integrates a Convolutional Neural Network (EfficientNet-B0) with a Flask-based web application to provide real-time tumor predictions and an interactive dashboard for model evaluation and dataset analysis.

This work was developed as part of a Capstone Project (MCA, CMR University, 2023–2025).

Features

Deep Learning Model (EfficientNet-B0)
Trained to classify MRI images into four categories:

glioma

meningioma

notumor

pituitary

Flask-Powered Application

Upload MRI images and run on-demand predictions.

Displays predicted class and confidence score.

Interactive Dashboard

Dataset distribution visualization

ROC curve

Confusion matrix

Classification report (precision, recall, F1-score)

Training history charts (accuracy & loss)

Artifacts Included

modelfinal.h5 — final trained model

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
Deep Learning

TensorFlow / Keras

EfficientNet-B0 transfer learning

NumPy, OpenCV, Pillow

Web Application

Flask

HTML5, CSS3, JavaScript, Bootstrap

Visualization

Matplotlib

Seaborn

JSON-based charts

Dataset

Publicly available Brain MRI dataset (e.g., Kaggle / BRATS-style collections)

~3,000–4,000 images

Four classes: glioma, meningioma, notumor, pituitary

Preprocessing pipeline:

Resize to 224×224

Normalize pixel values

Data augmentation (rotation, flip, zoom)

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

Create a file requirements.txt with the following suggested contents:

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


Then install:

pip install -r requirements.txt

Running the Application

Start the Flask server:

python main.py


The application will be available at:

http://127.0.0.1:5000/

Pages:

Landing Page: Upload MRI → Predict tumor class

Dashboard: View ROC curve, confusion matrix, dataset stats, reports

API Usage
Prediction Endpoint
POST /predict


Form Data:
file → MRI image (jpg/png)

Response Example:

{
  "prediction": "glioma",
  "confidence": 0.9873
}

Model Performance (Summary)

AUC: 0.99–1.00 across all classes

Accuracy: Very high across training & testing

Confusion Matrix: Strong diagonal values; minimal misclassification

Classification Report:

Precision: 0.93–0.99

Recall: 0.93–1.00

F1-Score: 0.94–0.99

Visualizations are available on the dashboard.

Training Workflow (Notebook)

The notebook (Brain-Tumor_Detection.ipynb) contains:

Dataset loading & preprocessing

EfficientNet-B0 model creation

Training with augmentation, callbacks (checkpoint, early stopping)

Evaluation (ROC, confusion matrix, report)

Saving model & metrics

Artifacts are exported as JSON/NumPy for dashboard use.

Future Enhancements

Integration with hospital PACS systems (DICOM workflow)

Deployment on cloud platforms (AWS/GCP/Azure)

Mobile-based inference app

Explainable AI using Grad-CAM heatmaps

Further multi-class segmentation (tumor boundary detection)

Acknowledgements

Guide: Prof. Aurangazeb Khan

Institution: CMR University, School of Science and Computer Studies

Open-source contributors (TensorFlow, Keras, Flask, Bootstrap)

License

This project was developed as part of an academic capstone.
For academic or research reuse, please cite appropriately and credit the author.
