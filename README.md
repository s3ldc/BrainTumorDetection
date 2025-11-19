# MRI Tumor Detection using Deep Learning

A deep learning–based MRI brain tumor classification system built using **EfficientNet-B0** and deployed through a **Flask web application**. The system enables real-time tumor prediction from MRI images and provides an analytical dashboard for evaluating model performance.

This project was developed as part of the **MCA Capstone (CMR University, 2023–2025).**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Training Workflow](#training-workflow)
- [Future Enhancements](#future-enhancements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Project Overview

This project implements an automated MRI tumor detection system using a deep learning classifier trained on brain MRI images. The trained model (`modelfinal.h5`) categorizes MRIs into:

- glioma  
- meningioma  
- notumor  
- pituitary  

A Flask web application allows users to upload MRI images for prediction and provides an interactive dashboard showing ROC curves, confusion matrices, classification reports, and dataset analytics.

---

## Features

- EfficientNet-B0–based deep learning classifier
- Real-time MRI tumor prediction via Flask app
- Dashboard visualizations:
  - Dataset distribution
  - ROC curve
  - Confusion matrix
  - Classification report
  - Training accuracy/loss curves
- Pre-trained model included (`modelfinal.h5`)
- JSON data exports for evaluation metrics

---

## Tech Stack

### Backend
- Python  
- Flask  
- TensorFlow / Keras  

### Frontend
- HTML5  
- CSS3  
- Bootstrap  
- JavaScript  

### Visualization
- Matplotlib  
- Seaborn  

---

## Dataset

- Public brain MRI datasets (Kaggle / BRATS-style)
- Classes: glioma, meningioma, notumor, pituitary
- Images resized to **224×224**
- Preprocessing:
  - Normalization
  - Augmentation (rotation, zoom, flip)
- Split:
  - 70% training
  - 20% validation
  - 10% testing

---

## Screenshots

### Homepage (MRI Upload & Detection)
![Homepage Screenshot](your-image-link-here)

### Model Evaluation Dashboard
![Dashboard Screenshot](your-image-link-here)

*(Replace links with your GitHub image URLs after uploading screenshots.)*

---

## Project Structure

```
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
```

---

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd CAPSTONE_PROJECT
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

Activate:
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt`:

```
flask
tensorflow
numpy
opencv-python
pillow
matplotlib
scikit-learn
pandas
seaborn
gunicorn
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Start the Flask server:
```bash
python main.py
```

### Access the application:
```
http://127.0.0.1:5000/
```

### Pages available:
- Homepage — Upload MRI & get prediction  
- Dashboard — Evaluation metrics, dataset analysis, charts  

---

## API Documentation

### POST /predict

Uploads an MRI image and returns a tumor prediction.

**Form Data:**
- `file`: MRI image (jpg/png)

**Sample Response:**

```json
{
  "prediction": "glioma",
  "confidence": 0.9873
}
```

---

## Model Performance

- AUC: 0.99–1.00  
- Precision: 0.93–0.99  
- Recall: 0.93–1.00  
- F1 Score: 0.94–0.99  

Visual outputs:
- `classification_report.json`
- `history.json`
- dashboard charts  

---

## Training Workflow

The Jupyter notebook includes:

1. Dataset loading  
2. Preprocessing & augmentation  
3. EfficientNet-B0 model creation  
4. Training with callbacks  
5. Evaluation (ROC, confusion matrix, reports)  
6. Saving the model and metrics  

---

## Future Enhancements

- Grad-CAM visual explanations  
- Hospital PACS (DICOM) Integration  
- Cloud Deployment (AWS, Render, GCP)  
- Lightweight mobile version  
- Tumor segmentation model  

---

## Acknowledgements

- Guide: Prof. Aurangazeb Khan  
- Institution: CMR University  
- TensorFlow, Flask, Bootstrap communities  
- MRI dataset contributors  

---

## License

This project was developed as part of an academic capstone.  
For reuse or modification, please credit the author.

