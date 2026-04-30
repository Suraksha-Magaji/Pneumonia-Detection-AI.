# 🫁 Pneumonia Detection AI

A deep learning model that analyzes chest X-ray images and detects pneumonia with **89% accuracy**.

## 🔴 Live Demo
👉 [Try it here](https://suru0219-pneumonia-scanner0-2.hf.space)

## 📊 Model Performance
- ✅ 89.4% Test Accuracy
- ✅ Trained on 5,216 chest X-ray images
- ✅ Automatically rejects non-X-ray images

## 🔧 Tech Stack
- Python
- TensorFlow & Keras
- MobileNetV2 (Transfer Learning)
- Gradio
- Hugging Face Spaces
- Kaggle GPU (Tesla T4)

## 🧠 How It Works
1. User uploads a chest X-ray image
2. Image is validated to ensure it is a real X-ray
3. MobileNetV2 model analyzes the image
4. Result is returned as NORMAL or PNEUMONIA with confidence score

## 📁 Project Structure
├── app.py               # Gradio web app
├── requirements.txt     # Dependencies
└── pneumonia_savedmodel # Trained TensorFlow model

## 🗂️ Dataset
Used the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.
- Training: 5,216 images
- Testing: 624 images
- Classes: NORMAL and PNEUMONIA

## ⚙️ How to Run Locally
```bash
pip install tensorflow gradio pillow numpy
python app.py
```

## 📜 License
MIT License
