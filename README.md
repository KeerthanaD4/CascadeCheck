# 📸 CascadeCheck

**CascadeCheck** is a Streamlit web application that evaluates the performance of a Haar Cascade classifier for face detection. It allows you to upload ZIP files of images that contain faces and those that do not, and then provides detailed accuracy metrics based on the detection results.

## 🚀Features
- Upload and extract ZIP files of face and non-face images.
- Automatically extracts and processes image data.
- Perform face detection using OpenCV’s Haar Cascade classifier.
- Calculates:
  ✅ Accuracy
  ✅ Precision
  ✅ Recall
  ✅ F1 Score
- Presents results in a clean, interactive Streamlit interface.

## 🖼️How It Works
1. Upload two ZIP files:
- One containing face images.
- One containing non-face images.

2. The app extracts and processes the images.

3. Face detection is performed using the haarcascade_frontalface_default.xml classifier.

4. Detection results are classified as:
- True Positive (TP): Faces detected in face images.
- False Negative (FN): Faces not detected in face images.
- True Negative (TN): No face detected in non-face images.
- False Positive (FP): Face detected in non-face images.

5. Accuracy metrics are calculated and displayed.

## 📊 Example Metrics Output
**Haar Cascade Face Detection Accuracy Report:**

✅True Positives (TP): 50

✅False Negatives (FN): 10

✅True Negatives (TN): 45

✅False Positives (FP): 5

✅Accuracy: 0.90

✅Precision: 0.91

✅Recall: 0.83

✅F1 Score: 0.87

## 🛠️ Tech Stack
* Streamlit
* OpenCV
* Python

## 🧪 Installation
1. Clone the repo:
```sh
git clone https://github.com/your-username/haar-face-detection-evaluation.git
cd haar-face-detection-evaluation
```

2. Install dependencies:
```sh
pip install streamlit opencv-python
```

3. Run the app:
```sh
streamlit run app.py
```

4. Upload ZIP Files
Prepare two zip files:
- One with face images
- One with non-face images
Upload them via the app UI and let CascadeCheck do the evaluation.

## 📄 License
The Haar cascade file is licensed under the Intel OpenCV license. All other code is © 2025 by you.
