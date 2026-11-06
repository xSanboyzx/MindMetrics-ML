"""Local Streamlit companion, using the same artifacts as the browser explorer."""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project.mindmetrics.inference import FEATURES, load_artifacts, predict  # noqa: E402

st.set_page_config(page_title="MindMetrics · Model explorer", page_icon="📊", layout="wide")
st.title("MindMetrics · Model explorer")
st.caption("Academic demonstration · synthetic data · session-only inputs")
st.write(
    "Explore patterns learned by a residual neural network from 3,500 synthetic records. "
    "These are model associations, not clinical assessments or predictions of personal wellbeing."
)


@st.cache_resource
def artifacts():
    return load_artifacts()


try:
    loaded = artifacts()
except (OSError, ValueError, RuntimeError, KeyError) as exc:
    st.error(f"Model setup failed: {exc}. Restore the committed model and scaler artifacts.")
    st.stop()

presets = json.loads((Path(__file__).parents[1] / "mindmetrics/presets.json").read_text())
chosen = st.selectbox("Fictional example", presets, format_func=lambda p: p["name"])
with st.form("profile"):
    values = {}
    columns = st.columns(3)
    for index, group in enumerate(dict.fromkeys(f["group"] for f in FEATURES)):
        with columns[index]:
            st.subheader(group)
            for i, feature in enumerate(FEATURES):
                if feature["group"] == group:
                    values[feature["key"]] = st.number_input(
                        f"{feature['label']} ({feature['unit']})",
                        min_value=float(feature["min"]),
                        max_value=float(feature["max"]),
                        value=float(chosen["values"][i]),
                        key=f"{chosen['id']}-{feature['key']}",
                    )
    submitted = st.form_submit_button("Compare predictions")
if submitted:
    adjusted = [values[f["key"]] for f in FEATURES]
    baseline, result = predict([chosen["values"], adjusted], loaded)
    left, right = st.columns(2)
    left.metric("Fictional baseline", f"{baseline:.2f}")
    right.metric(
        "Adjusted model output",
        f"{result:.2f}",
        delta=f"{result - baseline:+.2f}",
        delta_color="off",
    )
    if result < 0 or result > 10:
        st.warning(
            "Output falls outside the dataset's 0–10 target range. It is shown without clipping."
        )
st.info(
    "Controls use observed dataset ranges. Combining individually valid values can still produce "
    "a profile unlike the training data. Score labels are dataset variables, not clinical instruments."
)
