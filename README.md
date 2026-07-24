# 🎙️ Telehealth AI: Real-Time Patient Voice Urgency Triage

An end-to-end Deep Learning application designed to analyze vocal acoustics and speech transcript features to instantly score patient emergency triage levels in real-time. Built with TensorFlow/Keras, Streamlit, and Scikit-learn.

---

## 📌 Repository Overview & Project Structure

```text
.
├── app.py                            # Streamlit web application interface
├── model.ipynb                       # Jupyter notebook for EDA, feature engineering, and ANN training
├── model.h5                          # Trained Keras Artificial Neural Network (ANN) model
├── scaler.pkl                        # StandardScaler pickle file for input normalization
└── speech_recognition_transcription  # Dataset containing audio/speech acoustic features

Telehealth AI: Real-Time Patient Voice Urgency Triage is an advanced Deep Learning system powered by a 5-layer Artificial Neural Network (ANN) that analyzes vocal acoustics & speech features with 96.9% R² Accuracy to predict emergency patient urgency via web UI and live audio.
```
