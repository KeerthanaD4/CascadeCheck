import streamlit as st
import cv2
import os
import zipfile
import tempfile

# Set page config
st.set_page_config(page_title="Face Detection Accuracy", layout="centered")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces)

def extract_zip(uploaded_zip, extract_to):
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(extract_to)

def evaluate_model(face_dir, non_face_dir):
    TP = FN = TN = FP = 0

    for filename in os.listdir(face_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(face_dir, filename)
            TP += detect_faces(image_path) > 0
            FN += detect_faces(image_path) == 0

    for filename in os.listdir(non_face_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(non_face_dir, filename)
            TN += detect_faces(image_path) == 0
            FP += detect_faces(image_path) > 0

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return TP, FN, TN, FP, accuracy, precision, recall, f1

# --- UI ---
st.markdown("<h1 style='text-align: center; color: #3366cc;'>📸 Haar Cascade Face Detection Evaluation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload ZIP files of face and non-face images to evaluate detection accuracy.</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload section
col1, col2 = st.columns(2)
with col1:
    face_zip = st.file_uploader("⬆️ Upload ZIP of Face Images", type="zip", key="faces")
with col2:
    non_face_zip = st.file_uploader("⬆️ Upload ZIP of Non-Face Images", type="zip", key="nonfaces")

if face_zip and non_face_zip:
    with tempfile.TemporaryDirectory() as temp_dir:
        face_dir = os.path.join(temp_dir, "faces")
        non_face_dir = os.path.join(temp_dir, "non_faces")
        os.makedirs(face_dir, exist_ok=True)
        os.makedirs(non_face_dir, exist_ok=True)

        extract_zip(face_zip, face_dir)
        extract_zip(non_face_zip, non_face_dir)

        st.success("✅ Files extracted successfully. Running evaluation...")

        TP, FN, TN, FP, accuracy, precision, recall, f1 = evaluate_model(face_dir, non_face_dir)

        # Terminal-style output
        report = f"""
Haar Cascade Face Detection Accuracy Report
-------------------------------------------
True Positives (TP): {TP}
False Negatives (FN): {FN}
True Negatives (TN): {TN}
False Positives (FP): {FP}
Accuracy: {accuracy:.2f}
Precision: {precision:.2f}
Recall: {recall:.2f}
F1 Score: {f1:.2f}
"""
        st.subheader("📄 Detection Report")
        st.code(report, language="text")

        st.subheader("📊 Metrics Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("Precision", f"{precision:.2%}")
        col3.metric("Recall", f"{recall:.2%}")
        col4.metric("F1 Score", f"{f1:.2%}")
else:
    st.info("📂 Please upload both ZIP files to begin.")
