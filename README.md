# 🌟 MindMetrics-ML  
### Predicting Quality of Life Using Behavioral, Mental Health, and Digital Usage Data

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-MLP-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## Project Overview

MindMetrics-ML is a machine learning project that predicts an individual's **Quality of Life (QoL)** based on:

- Mental health indicators (anxiety, depression, stress)
- Digital behavior patterns (screen time, notifications, social media use)
- Daily lifestyle factors (sleep quality, physical activity, productivity)

Using a fully-connected neural network (MLP), the system learns patterns across 3,500 synthetic participant samples and generates a **QoL/Happiness score**.  
The trained model is deployed through a **Streamlit web app**, allowing users to interactively input lifestyle factors and receive a personalized prediction.

---

## Problem Statement

The rise of digital technologies has reshaped how individuals work, socialize, sleep, and cope with stress. Understanding how these behaviors influence mental health and overall wellbeing is challenging due to:

- Complex, nonlinear interactions  
- Multiple overlapping lifestyle factors  
- Noise and variability in human behavior  

This project solves the problem using **machine learning**, allowing a neural network to automatically discover patterns and estimate QoL from measurable features.

---

## Machine Learning Approach

### **Model Type:**  
Multilayer Perceptron (MLP) regression model using PyTorch.

### **Key Features Used:**
Examples include:

- `anxiety_score`
- `depression_score`
- `stress_level`
- `sleep_hours`
- `device_hours_per_day`
- `social_media_mins`
- `notifications_per_day`
- `digital_dependence_score`
- `productivity_score`
- …and more

### **Pipeline Summary:**
1. Data loading + preprocessing  
2. Train/validation/test split  
3. Normalization using StandardScaler  
4. Neural network training with Adam optimizer  
5. Model evaluation (MAE, RMSE, R²)  
6. Model + scaler saved for deployment  
7. Streamlit app loads model → takes user input → outputs QoL prediction  

---

## 📂 Project Structure
MindMetrics-ML/
│
├── data/
│ └── raw/
│ └── dataset.csv
│
├── notebooks/
│ ├── 01_eda.ipynb
│ └── 02_model_training.ipynb
│
├── deployment/
│ ├── app.py
│ └── model/
│ ├── best_model.pt
│ └── scaler.pkl
│
├── venv/ # Optional / local only
├── requirements.txt
└── README.md

---

# How to Run the Project

### **1️⃣ Clone the repository**
```bash
git clone https://github.com/<your-username>/MindMetrics-ML.git
cd MindMetrics-ML
```
### 2️⃣ **Create and activate a virtual environment**
Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

Mac/Linux:
python3 -m venv venv
source venv/bin/activate

### 3️⃣ **Install dependencies**
```bash 
pip install -r requirements.txt
```

### 4️⃣ **(Optional) Train the model from scratch**

Open:
```bash 
notebooks/02_model_training.ipynb
```

Running the notebook will:

- preprocess the dataset

- train the neural network

- evaluate performance

- export the trained model + scaler

Saved files appear in:
```bash 
deployment/model/
```

### **5️⃣ Run the Streamlit App**

From inside the deployment folder:
```bash 
python -m streamlit run app.py
```

If you encounter import errors, ensure your virtual environment is active.

This will launch the app at:
```bash 
http://localhost:8501
```

## License

This project was developed for academic use as part of a Machine Learning course.
You are free to modify and extend it.

## Acknowledgements

- Dataset: Digital Lifestyle Benchmark (Kaggle)

- Libraries: PyTorch, Streamlit, scikit-learn, NumPy, Pandas

- Course: CSCI 4050U — Machine Learning
