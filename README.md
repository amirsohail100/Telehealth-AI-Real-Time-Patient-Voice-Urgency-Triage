<div align="center">

# 🎙️ Telehealth AI: Real-Time Patient Voice Urgency Triage

### _Next-Generation Vocal Acoustic Analytics for Emergency Triage Scoring_

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Accuracy](https://img.shields.io/badge/Val_R²_Accuracy-96.90%25-brightgreen?style=for-the-badge)](#-model-accuracy--performance-metrics)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

---

### 📌 Table of Contents

- [📖 Executive Summary](#-executive-summary)
- [🖥️ Streamlit Web Application Interface](#️-streamlit-web-application-interface)
- [📂 Repository Directory Structure](#-repository-directory-structure)
- [🧠 ANN Architecture & Neuron Layout](#-ann-architecture--neuron-layout)
- [📊 Model Accuracy & Performance Metrics](#-model-accuracy--performance-metrics)
- [⚙️ How to Setup & Run](#️-how-to-setup--run)
- [👨‍💻 Author](#-author)

---

</div>

## 📖 Executive Summary

**Telehealth AI: Real-Time Patient Voice Urgency Triage** is an end-to-end Deep Learning application designed for emergency dispatchers and telehealth professionals. The platform extracts vocal acoustics and speech transcription features from patient calls to immediately evaluate urgency scores.

Powered by a customized **Artificial Neural Network (ANN)**, the system achieves **96.90% Validation $R^2$ Accuracy**, ensuring high reliability for emergency prioritization.

---

## 🖥️ Streamlit Web Application Interface

Below is the user interface designed for clinical workflows:

![Streamlit Web Application UI](image_5aa286.png)

> **Key UI Capabilities:**
>
> - **Dual Input Methods:** Toggle between pre-recorded audio files (`.wav`, `.mp3` up to 200MB) and **Live Microphone Input**.
> - **Real-Time Inference Engine:** Instant neural network scoring for patient calls.
> - **Sandbox / Model Status Monitor:** Clear visual indicators for model loading state and active execution modes.

---

## 📂 Repository Directory Structure

```text
.
├── app.py                            # Streamlit frontend & inference code
├── model.h5                          # Trained Keras Artificial Neural Network (ANN) model
├── model.ipynb                       # Jupyter Notebook (EDA, Preprocessing, ANN Training)
├── scaler.pkl                        # StandardScaler object for input normalization
├── speech_recognition_transcription  # Dataset containing audio/speech acoustic features
├── image_5aa286.png                  # UI Screenshot
└── image_5aa27f.png                  # Model Performance Plots

Telehealth AI: Real-Time Patient Voice Urgency Triage is an advanced Deep Learning system powered by a 5-layer Artificial Neural Network (ANN) that analyzes vocal acoustics & speech features with 96.9% R² Accuracy to predict emergency patient urgency via web UI and live audio.
```
