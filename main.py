from flask import Flask, render_template, request, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # render without GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json

app = Flask(__name__)

# ---- Model & Data ----
model = load_model('modelfinal.h5')

TRAIN_DIR = 'MRI Images/Training'
TEST_DIR  = 'MRI Images/Testing'
VALID_EXTS = ('.jpg', '.jpeg', '.png')

class_labels = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

# ---- Folders ----
UPLOAD_FOLDER = 'static/uploads'
CHARTS_FOLDER = 'static/charts'
DATA_FOLDER   = 'static/data'  # optional: where history.json/npz could live

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHARTS_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------- Helpers ----------
def preprocess_image(fp, image_size=224):
    img = load_img(fp, target_size=(image_size, image_size))
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    return arr

def list_images_with_labels(base_dir):
    paths, labels = [], []
    for label in sorted(os.listdir(base_dir)):
        d = os.path.join(base_dir, label)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith(VALID_EXTS):
                paths.append(os.path.join(d, fname))
                labels.append(label)
    return paths, labels

def open_images(paths):
    imgs = []
    for p in paths:
        try:
            imgs.append(preprocess_image(p))
        except Exception as e:
            print(f"⚠️ Failed to load {p}: {e}")
    return np.array(imgs)

def load_history():
    """
    Tries to read training history curves if you exported them from the notebook.
    - JSON: keys like 'sparse_categorical_accuracy', 'val_sparse_categorical_accuracy', 'loss', 'val_loss'
    - NPZ: same keys stored as arrays
    If not found, returns None and the training cards will be hidden.
    """
    json_path = os.path.join(DATA_FOLDER, 'history.json')
    npz_path  = os.path.join(DATA_FOLDER, 'history.npz')

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        return {k: data[k].tolist() for k in data.files}
    return None

def plot_training_curves(history):
    acc = history.get('sparse_categorical_accuracy') or history.get('accuracy')
    val_acc = history.get('val_sparse_categorical_accuracy') or history.get('val_accuracy')
    loss = history.get('loss')
    val_loss = history.get('val_loss')

    if not (acc and val_acc and loss and val_loss):
        return None  # incomplete history

    epochs = list(range(1, len(acc) + 1))

    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(True); plt.legend()
    acc_path = os.path.join(CHARTS_FOLDER, 'accuracy.png')
    plt.savefig(acc_path, bbox_inches='tight'); plt.close()

    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
    loss_path = os.path.join(CHARTS_FOLDER, 'loss.png')
    plt.savefig(loss_path, bbox_inches='tight'); plt.close()

    # Combined
    plt.figure(figsize=(6,4))
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Combined Accuracy & Loss')
    plt.xlabel('Epoch'); plt.ylabel('Value'); plt.grid(True); plt.legend()
    combined_path = os.path.join(CHARTS_FOLDER, 'combined.png')
    plt.savefig(combined_path, bbox_inches='tight'); plt.close()

    # Summary stats
    summary = {
        'best_train_acc': float(np.max(acc)),
        'best_val_acc': float(np.max(val_acc)),
        'lowest_train_loss': float(np.min(loss)),
        'lowest_val_loss': float(np.min(val_loss))
    }

    return {
        'accuracy': f'charts/accuracy.png',
        'loss': f'charts/loss.png',
        'combined': f'charts/combined.png',
        'summary': summary
    }

def compute_eval_and_plots():
    # ---- Load test set ----
    test_paths, test_labels = list_images_with_labels(TEST_DIR)
    if len(test_paths) == 0:
        raise RuntimeError(f"No test images found in {TEST_DIR}")

    X = open_images(test_paths)
    y_true = np.array([class_labels.index(lbl) for lbl in test_labels])

    # ---- Predict ----
    y_proba = model.predict(X, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)

    # ---- Classification report (dict) ----
    report = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0
    )

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    cm_path = os.path.join(CHARTS_FOLDER, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight'); plt.close()

    # ---- ROC curve (multi-class) ----
    # If only 2 classes, this still works (2 columns after binarize).
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_labels)))
    fpr, tpr, roc_auc = {}, {}, {}

    plt.figure(figsize=(7,6))
    for i in range(len(class_labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_labels[i]} (AUC={roc_auc[i]:.2f})')

    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.title('ROC Curve'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right'); plt.grid(True)
    roc_path = os.path.join(CHARTS_FOLDER, 'roc_curve.png')
    plt.savefig(roc_path, bbox_inches='tight'); plt.close()

    return {
        'charts': {
            'confusion_matrix': 'charts/confusion_matrix.png',
            'roc_curve': 'charts/roc_curve.png'
        },
        'report': report
    }

def plot_dataset_distribution():
    """Plot how many images per class are in Training and Testing sets."""
    counts = {"train": {}, "test": {}}

    for split, base_dir in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        for lbl in class_labels:
            d = os.path.join(base_dir, lbl)
            if os.path.exists(d):
                counts[split][lbl] = len([f for f in os.listdir(d) if f.lower().endswith(VALID_EXTS)])
            else:
                counts[split][lbl] = 0

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    width = 0.35
    x = np.arange(len(class_labels))

    train_counts = [counts["train"][lbl] for lbl in class_labels]
    test_counts  = [counts["test"][lbl] for lbl in class_labels]

    plt.bar(x - width/2, train_counts, width, label="Train")
    plt.bar(x + width/2, test_counts,  width, label="Test")

    plt.xticks(x, class_labels, rotation=45)
    plt.ylabel("Image Count")
    plt.title("Dataset Distribution")
    plt.legend()
    plt.tight_layout()

    dist_path = os.path.join(CHARTS_FOLDER, "dataset_distribution.png")
    plt.savefig(dist_path, bbox_inches="tight")
    plt.close()

    return "static/charts/dataset_distribution.png"

# ---------- Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            fp = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(fp)

            # Predict once for the uploaded image (home page behavior unchanged)
            img = preprocess_image(fp)
            img = np.expand_dims(img, axis=0)
            preds = model.predict(img, verbose=0)
            idx = int(np.argmax(preds, axis=1)[0])
            conf = float(np.max(preds, axis=1)[0])

            result = "No Tumor" if class_labels[idx].lower() == 'notumor' else f"Tumor: {class_labels[idx]}"
            return render_template(
                'index.html',
                result=result,
                confidence=f"{conf*100:.2f}%",
                file_path=url_for('get_uploaded_file', filename=file.filename)
            )
    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

import json

@app.route("/dashboard")
def dashboard():
    charts = {}
    summary = {}
    classification_report_data = {}
    history_available = False

    # --- Dataset Distribution ---
    charts['dataset_distribution'] = os.path.join("static", "charts", "dataset_distribution.png")

    # --- Training History ---
    history_path = "history.json"
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        history_available = True

        # Detect correct accuracy keys
        acc_key = None
        val_acc_key = None
        for key in history.keys():
            if "acc" in key.lower() and not key.lower().startswith("val"):
                acc_key = key
            if "val_acc" in key.lower() or "val_accuracy" in key.lower():
                val_acc_key = key

        # Plot Accuracy
        if "sparse_categorical_accuracy" in history and "val_sparse_categorical_accuracy" in history:
            plt.figure()
            plt.plot(history["sparse_categorical_accuracy"], label="Train Accuracy")
            plt.plot(history["val_sparse_categorical_accuracy"], label="Val Accuracy")
            plt.legend()
            plt.title("Training vs Validation Accuracy")
            acc_path = os.path.join("static", "charts", "accuracy.png")
            plt.savefig(acc_path)
            charts['accuracy'] = acc_path
            plt.close()
            print(f" Accuracy chart saved at: {acc_path}")
        else:
            print(" Accuracy keys not found in history.")

        # Plot Loss
        plt.figure()
        plt.plot(history["loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title("Training vs Validation Loss")
        loss_path = os.path.join("static", "charts", "loss.png")
        plt.savefig(loss_path)
        charts['loss'] = loss_path
        plt.close()

        # Combined chart
        plt.figure(figsize=(6, 4))
        if acc_key and val_acc_key:
            plt.plot(history[acc_key], label="Train Acc")
            plt.plot(history[val_acc_key], label="Val Acc")
        plt.plot(history["loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title("Combined Metrics")
        combined_path = os.path.join("static", "charts", "combined.png")
        plt.savefig(combined_path)
        charts['combined'] = combined_path
        plt.close()

        # Summary statistics
        summary = {
            "best_train_acc": max(history[acc_key]) if acc_key else 0,
            "best_val_acc": max(history[val_acc_key]) if val_acc_key else 0,
            "lowest_train_loss": min(history["loss"]),
            "lowest_val_loss": min(history["val_loss"]),
        }

    # --- Test Predictions ---
    preds_path = "test_predictions.npy"
    labels_path = "test_labels.npy"
    if os.path.exists(preds_path) and os.path.exists(labels_path):
        y_pred = np.load(preds_path)
        y_true = np.load(labels_path)

        # Convert logits to class predictions
        if y_pred.ndim > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = (y_pred > 0.5).astype(int)

        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

        class_names = sorted(os.listdir(os.path.join("MRI Images", "Testing")))
        classification_report_data = classification_report(
            y_true, y_pred_classes, target_names=class_names, output_dict=True
        )

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = os.path.join("static", "charts", "confusion_matrix.png")
        plt.savefig(cm_path)
        charts['confusion_matrix'] = cm_path
        plt.close()

        # ROC Curve
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # Multi-class ROC
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
            plt.figure()
            for i in range(len(class_names)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.legend()
            plt.title("ROC Curve")
        else:
            # Binary ROC
            fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1] if y_pred.ndim > 1 else y_pred)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.legend()
            plt.title("ROC Curve")

        roc_path = os.path.join("static", "charts", "roc_curve.png")
        plt.savefig(roc_path)
        charts['roc_curve'] = roc_path
        plt.close()

    return render_template(
        "dashboard.html",
        charts=charts,
        summary=summary,
        classification_report=classification_report_data,
        history_available=history_available
    )


if __name__ == '__main__':
    app.run(debug=True)
