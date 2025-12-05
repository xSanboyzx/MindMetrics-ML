import os
import numpy as np
import streamlit as st
import torch
from torch import nn
import joblib

# ---------- CONFIG ----------

FEATURE_COLS = [
    "age",
    "device_hours_per_day",
    "phone_unlocks",
    "notifications_per_day",
    "social_media_mins",
    "study_mins",
    "physical_activity_days",
    "sleep_hours",
    "sleep_quality",
    "anxiety_score",
    "depression_score",
    "stress_level",
    "focus_score",
    "productivity_score",
    "digital_dependence_score",
]

SCALER_PATH = os.path.join("model", "scaler.pkl")
MODEL_PATH = os.path.join("model", "best_model.pt")

# ---------- MODEL DEFINITION (must match training) ----------

class QoLRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@st.cache_resource
def load_artifacts():
    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Build model with correct input dim
    input_dim = len(FEATURE_COLS)
    model = QoLRegressor(input_dim)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return scaler, model


# ---------- STREAMLIT UI ----------

st.set_page_config(
    page_title="Digital Wellbeing / QoL Predictor",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Digital Wellbeing – Quality of Life Predictor")
st.write(
    "This demo uses a neural network trained on synthetic behavioral, mental health, "
    "and device usage data to **estimate a quality-of-life score**."
)
st.write(
    "_Note: This is an academic project, not a medical or clinical tool._"
)

scaler, model = load_artifacts()

st.header("1️⃣ Enter Lifestyle & Digital Behavior")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=13, max_value=80, value=22, step=1)
    device_hours_per_day = st.slider(
        "Device use per day (hours)", 0.0, 16.0, 6.0, 0.5
    )
    phone_unlocks = st.slider(
        "Phone unlocks per day", 0, 300, 80, 5
    )
    notifications_per_day = st.slider(
        "Notifications per day", 0, 500, 120, 10
    )
    social_media_mins = st.slider(
        "Social media time (minutes/day)", 0, 600, 120, 10
    )
    study_mins = st.slider(
        "Study / focused work (minutes/day)", 0, 600, 180, 10
    )
    physical_activity_days = st.slider(
        "Physical activity days per week", 0, 7, 3, 1
    )

with col2:
    sleep_hours = st.slider(
        "Sleep hours per night", 0.0, 14.0, 7.5, 0.5
    )
    sleep_quality = st.slider(
        "Sleep quality (1 = very poor, 10 = excellent)",
        1, 10, 7, 1
    )
    anxiety_score = st.slider(
        "Anxiety score (1–10)", 1, 10, 4, 1
    )
    depression_score = st.slider(
        "Depression score (1–10)", 1, 10, 3, 1
    )
    stress_level = st.slider(
        "Stress level (1–10)", 1, 10, 5, 1
    )
    focus_score = st.slider(
        "Focus score (1–10)", 1, 10, 7, 1
    )
    productivity_score = st.slider(
        "Productivity score (1–10)", 1, 10, 7, 1
    )
    digital_dependence_score = st.slider(
        "Digital dependence score (1 = very low, 10 = very high)",
        1, 10, 5, 1
    )

# Put inputs into the correct feature order
input_values = [
    age,
    device_hours_per_day,
    phone_unlocks,
    notifications_per_day,
    social_media_mins,
    study_mins,
    physical_activity_days,
    sleep_hours,
    sleep_quality,
    anxiety_score,
    depression_score,
    stress_level,
    focus_score,
    productivity_score,
    digital_dependence_score,
]

if st.button("🔮 Predict Quality of Life"):
    # Shape (1, num_features)
    x = np.array(input_values, dtype=np.float32).reshape(1, -1)

    # Scale using training-time scaler
    x_scaled = scaler.transform(x)

    # Convert to tensor and predict
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x_tensor).item()

    st.subheader("2️⃣ Model Prediction")
    st.metric(
        label="Predicted Quality of Life / Happiness Score",
        value=f"{pred:.2f}",
    )
    # Interpretation block
    st.write("### Interpretation (rough)")
    st.write(
        "- This score is on the same scale as the label you trained on "
        "(e.g., a happiness or QoL score from your dataset)."
    )
    st.write(
        "Lower values may suggest lower self-reported wellbeing, higher values suggest higher wellbeing, "
        "but exact meaning depends on how the original scores were defined."
    )

    # Little reflection block
    st.write("### What might influence this score?")
    st.write(
        "Try adjusting sleep hours, stress level, physical activity, and digital dependence to see "
        "how the prediction changes. This can give intuition about which factors the model has learned "
        "to associate with better or worse quality of life."
    )
else:
    st.info("Set the sliders above, then click **Predict Quality of Life** to see the model's output.")
