<div align="center">

# 🎙️ Telehealth AI: Real-Time Patient Voice Urgency Triage

### _Next-Generation Acoustic Analytics for Emergency Medical Triage & Clinical Decision Support_

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 Executive Summary

**Telehealth AI: Real-Time Patient Voice Urgency Triage** is an end-to-end Machine Learning solution designed to assist emergency dispatchers, telehealth triage nurses, and medical professionals. By analyzing subtle vocal acoustic properties and speech transcription features from patient calls, the system automatically evaluates patient distress levels and calculates real-time emergency urgency scores.

Built on a highly optimized, fully regularized Deep Neural Network (ANN), the application achieves exceptional predictive accuracy ($R^2 > 0.96$), ensuring critical cases receive instant clinical priority.

---

## 🎯 Key Features & Capabilities

- ⚡ **Dual Audio Input Pipeline:**
  - **Pre-recorded Audio Evaluation:** Upload `.wav` or `.mp3` patient recordings (up to 200MB).
  - **Live Microphone Stream:** Real-time speech input directly via web browser microphone.
- 🧠 **Multi-Stage Neural Inference Engine:** Custom deep ANN model trained on processed vocal acoustic extractions.
- 📈 **High Precision & Low Latency:** Optimized feature scaling delivering immediate inference results with minimal computational latency.
- 🎨 **Modern Interactive Dashboard:** Sleek, dark-themed Streamlit UI engineered for clinical usability.

---

## 🏗️ Repository Architecture & File Structure

```text
├── app.py                            # Streamlit web interface & audio processing pipeline
├── model.ipynb                       # End-to-end Jupyter Notebook (EDA, preprocessing, ANN training)
├── model.h5                          # Pre-trained Keras Deep Learning Model weights
├── scaler.pkl                        # Fitted StandardScaler object for input normalization
├── speech_recognition_transcription  # Dataset containing vocal acoustic and speech transcript features
└── README.md                         # Project documentation

Telehealth AI: Real-Time Patient Voice Urgency Triage is an advanced Deep Learning system powered by a 5-layer Artificial Neural Network (ANN) that analyzes vocal acoustics & speech features with 96.9% R² Accuracy to predict emergency patient urgency via web UI and live audio.
```
