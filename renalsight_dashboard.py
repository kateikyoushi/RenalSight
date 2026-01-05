import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import torch
import torch.nn as nn
from torchdiffeq import odeint
import urllib.parse

# Page configuration
st.set_page_config(
    page_title="DiaSight - Innovation in Diabetic Retinopathy",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== NEURAL ODE MODEL CLASSES ====================
class ODEFunc(nn.Module):
    """Neural network defining the ODE dynamics"""
    def __init__(self, n_features, hidden_dim=64):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_features)
        )

    def forward(self, t, x):
        return self.net(x)

class NeuralODEClassifier(nn.Module):
    """Complete Neural ODE classifier for trajectory prediction"""
    def __init__(self, input_dim, hidden_dim=64, ode_hidden_dim=64):
        super(NeuralODEClassifier, self).__init__()
        self.ode_func = ODEFunc(input_dim, hidden_dim=ode_hidden_dim)
        self.t = torch.tensor([0, 1], dtype=torch.float32)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        device = next(self.parameters()).device
        self.t = self.t.to(device)
        ode_out = odeint(self.ode_func, x, self.t, method='dopri5')
        x_evolved = ode_out[-1]
        logits = self.classifier(x_evolved)
        return logits

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_models():
    """Load trained models and preprocessors"""
    try:
        # CKD Stage Prediction Models
        with open('models/neurosymbolic_ckd.pkl', 'rb') as f:
            ns_model = pickle.load(f)
        if isinstance(ns_model, dict) and 'base_model' in ns_model:
            ns_model = ns_model['base_model']

        with open('models/xgboost_baseline.pkl', 'rb') as f:
            baseline_model = pickle.load(f)
        if isinstance(baseline_model, dict) and 'base_model' in baseline_model:
            baseline_model = baseline_model['base_model']

        with open('models/scaler_neurosymbolic.pkl', 'rb') as f:
            stage_scaler = pickle.load(f)
        with open('models/shap_explainer.pkl', 'rb') as f:
            stage_explainer = pickle.load(f)
        with open('config/model_metadata.json', 'r') as f:
            metadata = json.load(f)

        # Trajectory Prediction Models
        trajectory_model = None
        trajectory_scaler = None
        trajectory_features = None

        try:
            # Load Neural ODE model
            device = torch.device('cpu')  # Use CPU for Streamlit
            checkpoint = torch.load('models/neural_ode_model.pth', map_location=device, weights_only=False)

            with open('models/feature_list.pkl', 'rb') as f:
                trajectory_features = pickle.load(f)

            input_dim = len(trajectory_features)
            trajectory_model = NeuralODEClassifier(
                input_dim=input_dim,
                hidden_dim=checkpoint.get('hidden_dim', 64),
                ode_hidden_dim=checkpoint.get('ode_hidden_dim', 64)
            )
            trajectory_model.load_state_dict(checkpoint['model_state_dict'])
            trajectory_model.to(device)
            trajectory_model.eval()

            with open('models/neural_ode_scaler.pkl', 'rb') as f:
                trajectory_scaler = pickle.load(f)

            # st.success("✅ Trajectory prediction available")  # Removed sidebar usage
        except Exception as e:
            # st.warning(f"⚠️ Trajectory model not available: {str(e)[:50]}")  # Removed sidebar usage
            pass

        return ns_model, baseline_model, stage_scaler, stage_explainer, metadata, \
               trajectory_model, trajectory_scaler, trajectory_features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None, None, None

@st.cache_data
def load_data():
    """Load datasets"""
    try:
        test_predictions = pd.read_csv('data/test_predictions_with_confidence.csv')
        model_comparison = pd.read_csv('data/model_comparison_table.csv')
        yakap_clinics = pd.read_csv('data/YAKAP_CLINICS.csv')
        return test_predictions, model_comparison, yakap_clinics
    except Exception as e:
        st.warning(f"Could not load data files: {e}")
        return None, None, None

# ==================== CONSTANTS ====================
RISK_LEVELS = {
    1: {"label": "Stage 1 - Minimal Risk", "color": "#28a745", "desc": "Kidney damage with normal function"},
    2: {"label": "Stage 2 - Mild Risk", "color": "#90EE90", "desc": "Mildly reduced kidney function"},
    3: {"label": "Stage 3 - Moderate Risk", "color": "#ffc107", "desc": "Moderately reduced kidney function"},
    4: {"label": "Stage 4 - High Risk", "color": "#fd7e14", "desc": "Severely reduced kidney function"},
    5: {"label": "Stage 5 - Critical Risk", "color": "#dc3545", "desc": "Kidney failure - dialysis may be needed"}
}

TRAJECTORY_INFO = {
    0: {
        "label": "Stable Trajectory",
        "color": "#28a745",
        "desc": "Slow decline rate (<1 mL/min/year)",
        "icon": "📊",
        "prognosis": "Good prognosis - kidney function declining slowly"
    },
    1: {
        "label": "Progressive Trajectory", 
        "color": "#dc3545",
        "desc": "Faster decline rate (≥1 mL/min/year)",
        "icon": "📉",
        "prognosis": "Requires close monitoring and intervention"
    }
}

# ==================== CLINICAL RULES ====================
def check_clinical_rules(patient_data):
    """Check clinical rule violations"""
    warnings = []
    recommendations = []

    bun = patient_data['bun']
    urea = patient_data['urea']
    uacr = patient_data['UACR']
    alb = patient_data['ALB']
    hba1c = patient_data['hba1c']
    duration = patient_data['duration']

    if uacr >= 300:
        warnings.append("⚠️ Severe albuminuria detected (UACR ≥300 mg/g)")
        recommendations.append("Consider nephrology referral and RAAS inhibitor therapy")

    if bun > 60 or urea > 120:
        warnings.append("⚠️ Elevated waste products indicate advanced kidney dysfunction")
        recommendations.append("Urgent nephrology consultation recommended")

    if alb < 3.0:
        warnings.append("⚠️ Low albumin level may indicate malnutrition or kidney disease")
        recommendations.append("Nutritional assessment and dietary counseling")

    if hba1c > 9.0:
        warnings.append("⚠️ Poor glycemic control increases CKD progression risk")
        recommendations.append("Intensify diabetes management")

    if duration >= 15 and uacr > 30:
        recommendations.append("Long diabetes duration with albuminuria - monitor closely")

    return warnings, recommendations

def apply_soft_clinical_guidance(y_pred, y_pred_proba, X_data, confidence_scores, rule_weight=0.3):
    """Apply soft clinical rule guidance"""
    y_adjusted = y_pred.copy()
    y_proba_adjusted = y_pred_proba.copy()

    for idx in range(len(y_pred)):
        bun = X_data.iloc[idx]['bun']
        urea = X_data.iloc[idx]['urea']
        uacr = X_data.iloc[idx]['UACR']
        confidence = confidence_scores[idx]

        if confidence < 0.6:
            if bun > 60 or urea > 120:
                y_proba_adjusted[idx, 3:5] += rule_weight * 0.1
                y_proba_adjusted[idx, 0:2] -= rule_weight * 0.05

            if uacr >= 300:
                y_proba_adjusted[idx, 2:5] += rule_weight * 0.1
                y_proba_adjusted[idx, 0] -= rule_weight * 0.1

            if bun < 20 and urea < 40 and uacr < 30:
                y_proba_adjusted[idx, 0:2] += rule_weight * 0.1
                y_proba_adjusted[idx, 3:5] -= rule_weight * 0.1

            y_proba_adjusted[idx] = y_proba_adjusted[idx] / y_proba_adjusted[idx].sum()
            new_pred = np.argmax(y_proba_adjusted[idx]) + 1
            y_adjusted[idx] = new_pred

    return y_adjusted, y_proba_adjusted

# ==================== PREDICTION FUNCTIONS ====================
def predict_ckd_risk(patient_data, scaler, model, model_type="Neuro-Symbolic"):
    """Make CKD stage risk prediction"""
    features = ['age', 'sex', 'duration', 'hbp', 'sbp', 'dbp', 'hba1c', 'bun', 'urea', 'ALB', 'UACR']
    numerical_features = ['age', 'duration', 'sbp', 'dbp', 'hba1c', 'bun', 'urea', 'ALB', 'UACR']

    X = pd.DataFrame([patient_data])
    X_scaled = X.copy()
    X_scaled[numerical_features] = scaler.transform(X[numerical_features])

    pred = model.predict(X_scaled[features])[0]
    proba = model.predict_proba(X_scaled[features])[0]
    confidence = np.max(proba) * 100

    if model_type == "Neuro-Symbolic" and confidence < 60.0:
        adjusted_pred, adjusted_proba = apply_soft_clinical_guidance(
            np.array([pred]), 
            np.array([proba]), 
            X, 
            np.array([confidence / 100.0]),
            rule_weight=0.3
        )
        pred = adjusted_pred[0]
        proba = adjusted_proba[0]
        confidence = np.max(proba) * 100

    pred = max(1, min(5, int(pred)))
    return int(pred), confidence, proba

def predict_trajectory(patient_data, model, scaler, features):
    """Predict CKD progression trajectory using Neural ODE"""
    try:
        # Prepare input data
        X = pd.DataFrame([patient_data])
        X = X[features]  # Ensure correct feature order
        X_scaled = scaler.transform(X)

        # Convert to tensor
        device = next(model.parameters()).device
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1)

        pred_class = int(pred.cpu().numpy()[0])
        probs_np = probs.cpu().numpy()[0]
        confidence = float(np.max(probs_np) * 100)

        # Estimate decline rate based on current eGFR and trajectory
        egfr = patient_data['eGFR']
        age = patient_data['age']
        duration = patient_data['duration']

        # Estimate baseline eGFR
        baseline_egfr = 125 - max(0, age - 40)

        # Calculate observed decline rate
        if duration > 0:
            observed_decline = (baseline_egfr - egfr) / duration
        else:
            observed_decline = 0

        # Estimate future decline rate based on prediction
        if pred_class == 0:  # Stable
            estimated_decline_rate = min(observed_decline, 1.0)
        else:  # Progressive
            estimated_decline_rate = max(observed_decline, 3.0)

        # Calculate time to dialysis (eGFR < 15)
        if egfr > 15 and estimated_decline_rate > 0:
            years_to_dialysis = (egfr - 15) / estimated_decline_rate
        else:
            years_to_dialysis = None

        return {
            'prediction': pred_class,
            'confidence': confidence,
            'probabilities': probs_np,
            'decline_rate': estimated_decline_rate,
            'years_to_dialysis': years_to_dialysis,
            'current_egfr': egfr,
            'baseline_egfr': baseline_egfr
        }
    except Exception as e:
        st.error(f"Trajectory prediction error: {e}")
        return None

# ==================== VISUALIZATION FUNCTIONS ====================
def create_risk_gauge(prediction, confidence):
    """Create gauge chart for CKD stage risk"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"CKD Risk Stage<br><span style='font-size:0.8em'>Confidence: {confidence:.1f}%</span>"},
        gauge={
            'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': RISK_LEVELS[prediction]['color']},
            'steps': [
                {'range': [0, 1], 'color': '#e8f5e9'},
                {'range': [1, 2], 'color': '#c8e6c9'},
                {'range': [2, 3], 'color': '#fff9c4'},
                {'range': [3, 4], 'color': '#ffe0b2'},
                {'range': [4, 5], 'color': '#ffcdd2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 4
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig

def create_probability_chart(proba):
    """Create probability distribution chart for stage"""
    stages = [f"Stage {i}" for i in range(1, 6)]
    fig = go.Figure(data=[
        go.Bar(x=stages, y=proba * 100, marker_color=[RISK_LEVELS[i+1]['color'] for i in range(5)])
    ])
    fig.update_layout(
        title="Prediction Probability Distribution",
        xaxis_title="CKD Stage",
        yaxis_title="Probability (%)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_trajectory_gauge(prediction, confidence):
    """Create gauge for trajectory prediction"""
    traj_info = TRAJECTORY_INFO[prediction]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Trajectory Prediction<br><span style='font-size:0.8em'>Confidence: {confidence:.1f}%</span>"},
        gauge={
            'axis': {'range': [None, 1], 'tickvals': [0, 1], 'ticktext': ['Stable', 'Progressive']},
            'bar': {'color': traj_info['color']},
            'steps': [
                {'range': [0, 0.5], 'color': '#c8e6c9'},
                {'range': [0.5, 1], 'color': '#ffcdd2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig

def create_egfr_trajectory_plot(current_egfr, baseline_egfr, decline_rate, years_to_dialysis):
    """Create projected eGFR trajectory plot"""
    # Project future eGFR
    years_ahead = min(20, years_to_dialysis + 5 if years_to_dialysis else 20)
    time_points = np.linspace(0, years_ahead, 100)
    projected_egfr = current_egfr - (decline_rate * time_points)
    projected_egfr = np.maximum(projected_egfr, 0)  # Can't go below 0

    fig = go.Figure()

    # Projected trajectory
    fig.add_trace(go.Scatter(
        x=time_points,
        y=projected_egfr,
        mode='lines',
        name='Projected Trajectory',
        line=dict(color='#dc3545', width=3)
    ))

    # Current point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[current_egfr],
        mode='markers',
        name='Current eGFR',
        marker=dict(size=12, color='#007bff')
    ))

    # Dialysis threshold
    fig.add_hline(y=15, line_dash="dash", line_color="red", 
                  annotation_text="Dialysis Threshold (eGFR=15)")

    # CKD stage zones
    fig.add_hrect(y0=90, y1=150, fillcolor="green", opacity=0.1, 
                  annotation_text="Stage 1-2", annotation_position="right")
    fig.add_hrect(y0=60, y1=90, fillcolor="yellow", opacity=0.1,
                  annotation_text="Stage 3a", annotation_position="right")
    fig.add_hrect(y0=30, y1=60, fillcolor="orange", opacity=0.1,
                  annotation_text="Stage 3b-4", annotation_position="right")
    fig.add_hrect(y0=0, y1=30, fillcolor="red", opacity=0.1,
                  annotation_text="Stage 4-5", annotation_position="right")

    fig.update_layout(
        title="Projected eGFR Trajectory",
        xaxis_title="Years from Now",
        yaxis_title="eGFR (mL/min/1.73m²)",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )

    return fig

def display_shap_explanation(patient_data, scaler, model, explainer, pred):
    """Display SHAP explanation for stage prediction"""
    features = ['age', 'sex', 'duration', 'hbp', 'sbp', 'dbp', 'hba1c', 'bun', 'urea', 'ALB', 'UACR']
    numerical_features = ['age', 'duration', 'sbp', 'dbp', 'hba1c', 'bun', 'urea', 'ALB', 'UACR']

    X = pd.DataFrame([patient_data])
    X_scaled = X.copy()
    X_scaled[numerical_features] = scaler.transform(X[numerical_features])

    try:
        shap_values = explainer.shap_values(X_scaled[features])
        pred_class = pred - 1  # 0-based index

        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # List of arrays for each class
            values = shap_values[pred_class][0]
            base_values = explainer.expected_value[pred_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            # 2D array: (n_features, n_classes)
            values = shap_values[:, pred_class]
            base_values = explainer.expected_value[pred_class] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap.Explanation(
            values=values,
            base_values=base_values,
            data=X_scaled[features].values[0],
            feature_names=features
        ), show=False)
        st.pyplot(fig, bbox_inches='tight')
        plt.close()
    except Exception as e:
        st.warning(f"SHAP explanation not available: {e}")

# ==================== MAIN APPLICATION ====================
def main():
    st.title("🩺 RenalSight")
    st.subheader("AI-Powered Chronic Kidney Disease Risk Assessment System by DiaSight")

    # Load models
    ns_model, baseline_model, stage_scaler, stage_explainer, metadata, \
        trajectory_model, trajectory_scaler, trajectory_features = load_models()

    test_predictions, model_comparison, yakap_clinics = load_data()

    if ns_model is None:
        st.error("Failed to load models. Please ensure all model files are present.")
        return

    # Tab-based navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 Patient Assessment",
        "🏥 Referral Directory", 
        "📊 Model Performance",
        "ℹ️ About"
    ])

    with tab1:
        patient_assessment_page(ns_model, baseline_model, stage_scaler, stage_explainer,
                               trajectory_model, trajectory_scaler, trajectory_features)
    
    with tab2:
        referral_directory_page(yakap_clinics)
    
    with tab3:
        model_performance_page(test_predictions, model_comparison, metadata)
    
    with tab4:
        about_page(metadata)

def patient_assessment_page(ns_model, baseline_model, stage_scaler, stage_explainer,
                           trajectory_model, trajectory_scaler, trajectory_features):
    """Patient assessment with stage and trajectory prediction"""
    st.header("🏥 Patient Risk Assessment")
    st.markdown("Enter patient details below to assess CKD risk using our AI models.")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Patient Information")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=60, step=1)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 2)], format_func=lambda x: x[0])
        duration = st.number_input("Diabetes Duration (years)", min_value=0, max_value=60, value=10, step=1)
        hbp = st.selectbox("Hypertension", options=[("No", 1), ("Yes", 2)], format_func=lambda x: x[0])

        st.divider()

        st.subheader("❤️ Vital Signs")
        sbp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=220, value=130, step=1)
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=130, value=80, step=1)

        st.divider()

        st.subheader("🧪 Laboratory Results")
        hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=7.5, step=0.1)
        bun = st.number_input("BUN (mg/dL)", min_value=2.0, max_value=150.0, value=15.0, step=0.1)
        urea = st.number_input("Urea (mg/dL)", min_value=1.0, max_value=400.0, value=70.0, step=0.1)
        alb = st.number_input("Serum Albumin (g/dL)", min_value=1.0, max_value=6.0, value=4.0, step=0.1)
        uacr = st.number_input("UACR (mg/g)", min_value=0.0, max_value=4000.0, value=30.0, step=1.0)
        egfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=5.0, max_value=150.0, value=90.0, step=0.1,
                               help="Current estimated Glomerular Filtration Rate")

    with col2:
        st.subheader("🔍 Analysis & Results")

        # Create tabs for different predictions
        tab1, tab2 = st.tabs(["📊 Stage Assessment", "📉 Trajectory Prediction"])

        with tab1:
            if st.button("🔍 Analyze CKD Stage", type="primary", use_container_width=True, key="stage_btn"):
                # Prepare patient data (without eGFR for stage prediction)
                patient_data = {
                    'age': age, 'sex': sex[1], 'duration': duration, 'hbp': hbp[1],
                    'sbp': sbp, 'dbp': dbp, 'hba1c': hba1c,
                    'bun': bun, 'urea': urea, 'ALB': alb, 'UACR': uacr
                }

                # Make prediction
                pred, conf, proba = predict_ckd_risk(patient_data, stage_scaler, ns_model, "Neuro-Symbolic")

                # Display risk level
                risk_info = RISK_LEVELS[pred]
                st.markdown(f"""
                    <div style="background-color: {risk_info['color']}; padding: 20px; border-radius: 10px; 
                                margin: 10px 0; text-align: center; font-size: 1.5rem; font-weight: bold; color: white;">
                        {risk_info['label']}
                    </div>
                    <p style="text-align: center; font-size: 1.1rem; margin-top: 10px;">
                        {risk_info['desc']}
                    </p>
                """, unsafe_allow_html=True)

                st.divider()

                # Visualizations
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(create_risk_gauge(pred, conf), use_container_width=True)
                with col_b:
                    st.plotly_chart(create_probability_chart(proba), use_container_width=True)

                st.divider()

                # Clinical alerts
                warnings, recommendations = check_clinical_rules(patient_data)

                if warnings:
                    st.subheader("⚠️ Clinical Alerts")
                    for warning in warnings:
                        st.warning(warning)

                if recommendations:
                    st.subheader("💡 Recommendations")
                    for rec in recommendations:
                        st.info(rec)

                st.divider()

                # Biomarkers
                st.subheader("🔬 Key Biomarker Analysis")
                bio_col1, bio_col2, bio_col3, bio_col4 = st.columns(4)

                with bio_col1:
                    st.metric("BUN", f"{bun:.1f} mg/dL", 
                             "High" if bun > 20 else "Normal", delta_color="inverse")
                with bio_col2:
                    st.metric("UACR", f"{uacr:.1f} mg/g",
                             "High" if uacr > 300 else ("Moderate" if uacr > 30 else "Normal"),
                             delta_color="inverse")
                with bio_col3:
                    st.metric("HbA1c", f"{hba1c:.1f}%",
                             "Poor" if hba1c > 9 else ("Fair" if hba1c > 7 else "Good"),
                             delta_color="inverse")
                with bio_col4:
                    st.metric("Albumin", f"{alb:.1f} g/dL",
                             "Low" if alb < 3.5 else "Normal", delta_color="inverse")

                # SHAP
                with st.expander("🔬 View Detailed Feature Analysis (SHAP)"):
                    st.subheader("Feature Importance Analysis")
                    try:
                        st.image("viz/shap_summary_plot.png", caption="SHAP Feature Importance Summary", use_container_width=True)
                    except:
                        st.info("SHAP summary plot not available")

        with tab2:
            if trajectory_model is None:
                st.warning("⚠️ Trajectory prediction model not available. Please ensure neural_ode_model.pth is present.")
            else:
                if st.button("📉 Predict Progression Trajectory", type="primary", use_container_width=True, key="traj_btn"):
                    # Prepare patient data (WITH eGFR for trajectory)
                    patient_data_traj = {
                        'eGFR': egfr, 'UACR': uacr, 'age': age, 'sex': sex[1],
                        'duration': duration, 'hbp': hbp[1], 'sbp': sbp, 'dbp': dbp,
                        'hba1c': hba1c, 'bun': bun, 'urea': urea, 'ALB': alb
                    }

                    # Make trajectory prediction
                    traj_result = predict_trajectory(patient_data_traj, trajectory_model, 
                                                    trajectory_scaler, trajectory_features)

                    if traj_result:
                        pred_traj = traj_result['prediction']
                        conf_traj = traj_result['confidence']
                        traj_info = TRAJECTORY_INFO[pred_traj]

                        # Display trajectory prediction
                        st.markdown(f"""
                            <div style="background-color: {traj_info['color']}; padding: 20px; border-radius: 10px; 
                                        margin: 10px 0; text-align: center; font-size: 1.5rem; font-weight: bold; color: white;">
                                {traj_info['icon']} {traj_info['label']}
                            </div>
                            <p style="text-align: center; font-size: 1.1rem; margin-top: 10px;">
                                {traj_info['desc']}<br>
                                <strong>{traj_info['prognosis']}</strong>
                            </p>
                        """, unsafe_allow_html=True)

                        st.divider()

                        # Trajectory metrics
                        met_col1, met_col2, met_col3, met_col4 = st.columns(4)

                        with met_col1:
                            st.metric("Confidence", f"{conf_traj:.1f}%")
                        with met_col2:
                            st.metric("Decline Rate", f"{traj_result['decline_rate']:.2f} mL/min/yr",
                                     delta="Rapid" if traj_result['decline_rate'] > 3 else "Slow")
                        with met_col3:
                            st.metric("Current eGFR", f"{egfr:.1f}")
                        with met_col4:
                            if traj_result['years_to_dialysis']:
                                st.metric("Est. Time to Dialysis", 
                                         f"{traj_result['years_to_dialysis']:.1f} years")
                            else:
                                st.metric("Time to Dialysis", "N/A")

                        st.divider()

                        # Trajectory visualization
                        st.subheader("📈 Projected eGFR Trajectory")
                        if traj_result['years_to_dialysis']:
                            fig_traj = create_egfr_trajectory_plot(
                                traj_result['current_egfr'],
                                traj_result['baseline_egfr'],
                                traj_result['decline_rate'],
                                traj_result['years_to_dialysis']
                            )
                            st.plotly_chart(fig_traj, use_container_width=True)
                        else:
                            st.info("Unable to project trajectory - insufficient data")

                        st.divider()

                        # Trajectory-specific recommendations
                        st.subheader("💡 Trajectory-Based Recommendations")

                        if pred_traj == 1:  # Progressive
                            st.error("**Rapid Progression Detected**")
                            st.markdown("""
                            - **Urgent Action**: Schedule nephrology consultation within 2-4 weeks
                            - **Frequent Monitoring**: eGFR and UACR every 3 months
                            - **Intensify Treatment**: Optimize RAAS blockade, glycemic control
                            - **Lifestyle**: Strict blood pressure control, low-protein diet
                            - **Patient Education**: Prepare for potential renal replacement therapy
                            """)
                        else:  # Stable
                            st.success("**Stable Progression**")
                            st.markdown("""
                            - **Routine Monitoring**: Annual eGFR and UACR checks
                            - **Maintain Control**: Continue current management strategies
                            - **Prevention**: Emphasize glycemic and blood pressure control
                            - **Lifestyle**: Healthy diet, regular exercise, avoid nephrotoxins
                            - **Follow-up**: Schedule 6-12 month review
                            """)

                        # Probability breakdown
                        with st.expander("📊 View Probability Breakdown"):
                            prob_col1, prob_col2 = st.columns(2)
                            with prob_col1:
                                st.metric("Stable Probability", f"{traj_result['probabilities'][0]*100:.1f}%")
                            with prob_col2:
                                st.metric("Progressive Probability", f"{traj_result['probabilities'][1]*100:.1f}%")

def referral_directory_page(yakap_clinics):
    """Display YAKAP Clinics Referral Directory with pagination"""
    st.header("🏥 Referral Directory")
    st.markdown("YAKAP Clinics Directory for CKD Patient Referrals")

    st.divider()

    if yakap_clinics is None or yakap_clinics.empty:
        st.error("Unable to load YAKAP clinics data. Please ensure the data file is available.")
        return

    # Search and filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("🔍 Search clinics", placeholder="Enter clinic name or location...")
    
    with col2:
        municipality_filter = st.selectbox(
            "📍 Filter by Municipality",
            options=["All"] + sorted(yakap_clinics['Municipality'].dropna().unique().tolist())
        )
    
    with col3:
        facility_type_filter = st.selectbox(
            "🏥 Filter by Sector",
            options=["All"] + sorted(yakap_clinics['Sector of health facility'].dropna().unique().tolist())
        )

    # Apply filters
    filtered_clinics = yakap_clinics.copy()
    
    if search_term:
        filtered_clinics = filtered_clinics[
            filtered_clinics['Name of health facility'].str.contains(search_term, case=False, na=False) |
            filtered_clinics['Municipality'].str.contains(search_term, case=False, na=False) |
            filtered_clinics['Street Address'].str.contains(search_term, case=False, na=False)
        ]
    
    if municipality_filter != "All":
        filtered_clinics = filtered_clinics[filtered_clinics['Municipality'] == municipality_filter]
    
    if facility_type_filter != "All":
        filtered_clinics = filtered_clinics[filtered_clinics['Sector of health facility'] == facility_type_filter]

    # Pagination setup
    clinics_per_page = 50
    total_clinics = len(filtered_clinics)
    total_pages = (total_clinics + clinics_per_page - 1) // clinics_per_page  # Ceiling division

    if total_pages > 1:
        page = st.selectbox(
            "📄 Select Page",
            options=list(range(1, total_pages + 1)),
            format_func=lambda x: f"Page {x} of {total_pages}"
        )
        start_idx = (page - 1) * clinics_per_page
        end_idx = min(start_idx + clinics_per_page, total_clinics)
        clinics_to_show = filtered_clinics.iloc[start_idx:end_idx]
    else:
        clinics_to_show = filtered_clinics
        page = 1

    st.markdown(f"**Showing {len(clinics_to_show)} of {total_clinics} clinics**")

    # Display clinics
    if not clinics_to_show.empty:
        for idx, clinic in clinics_to_show.iterrows():
            with st.expander(f"🏥 {clinic.get('Name of health facility', 'N/A')} - {clinic.get('Municipality', 'N/A')}"):
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.markdown("**📍 Address:**")
                    st.write(f"{clinic.get('Street Address', 'N/A')}, {clinic.get('Municipality', 'N/A')}")
                    
                    st.markdown("**🏥 Sector:**")
                    st.write(clinic.get('Sector of health facility', 'N/A'))
                    
                    if pd.notna(clinic.get('Head of the facility')):
                        st.markdown("**👨‍⚕️ Head of Facility:**")
                        st.write(clinic.get('Head of the facility', 'N/A'))
                    
                    if pd.notna(clinic.get('Email Address')):
                        st.markdown("**📧 Email:**")
                        st.write(clinic.get('Email Address', 'N/A'))
                        
                        # Generate referral email link
                        email_subject = f"CKD Patient Referral - {clinic.get('Name of health facility', 'Clinic')}"
                        email_body = f"""Dear {clinic.get('Head of the facility', 'Sir/Madam')},

I am referring a CKD patient for specialized care and evaluation.

Please find the patient details and clinical information attached.

Facility: {clinic.get('Name of health facility', 'N/A')}
Address: {clinic.get('Street Address', 'N/A')}, {clinic.get('Municipality', 'N/A')}

Thank you for your attention to this referral.

Best regards,
RenalSight CKD Assessment System"""
                        
                        encoded_subject = urllib.parse.quote(email_subject)
                        encoded_body = urllib.parse.quote(email_body)
                        mailto_link = f"mailto:{clinic.get('Email Address')}?subject={encoded_subject}&body={encoded_body}"
                        
                        st.markdown(f"**[📤 Send Referral Email]({mailto_link})**")
                
                with col_b:
                    st.markdown("**📞 Contact Information:**")
                    if pd.notna(clinic.get('Telephone number')):
                        st.write(f"📱 {clinic.get('Telephone number')}")
                    else:
                        st.write("📱 Not available")
                    
                    st.markdown("**🌐 Services:**")
                    st.write("• CKD Management")
                    st.write("• Nephrology Consultation")
                    st.write("• Dialysis Services")
                    st.write("• Laboratory Testing")
    else:
        st.info("No clinics found matching your search criteria. Try adjusting your filters.")

    st.divider()
    st.markdown("**💡 Usage Notes:**")
    st.markdown("""
    - Use the search bar to find clinics by name, location, or address
    - Filter by municipality or facility type to narrow down options
    - Click on clinic names to expand and view detailed information
    - Use the referral email link to send patient referrals directly
    - All clinics listed are verified YAKAP partner facilities
    """)

def model_performance_page(test_predictions, model_comparison, metadata):
    """Display model performance metrics"""
    st.header("📊 Model Performance Metrics")
    st.markdown("Comprehensive evaluation of AI models on test data.")

    st.divider()

    if model_comparison is not None:
        st.subheader("📈 Model Comparison")
        st.dataframe(model_comparison, use_container_width=True)
        st.divider()

        # Extract Neuro-Symbolic performance metrics
        def get_metric_value(metric_name):
            """Get metric value from model_comparison dataframe"""
            if model_comparison is not None and 'Metric' in model_comparison.columns and 'Neuro-Symbolic' in model_comparison.columns:
                row = model_comparison[model_comparison['Metric'] == metric_name]
                if not row.empty:
                    return float(row['Neuro-Symbolic'].iloc[0])
            return 0.0

        st.subheader("🎯 Neuro-Symbolic Model Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            accuracy = get_metric_value('Test Accuracy')
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col2:
            balanced_acc = get_metric_value('Balanced Accuracy')
            st.metric("Balanced Accuracy", f"{balanced_acc:.2%}")
        with col3:
            f1_score = get_metric_value('Macro F1-Score')
            st.metric("Macro F1-Score", f"{f1_score:.3f}")
        with col4:
            kappa = get_metric_value('Cohen\'s Kappa')
            st.metric("Cohen's Kappa", f"{kappa:.3f}")

        st.divider()

    st.subheader("📊 Performance Visualizations")

    viz_cols = st.columns(2)

    viz_files = [
        ("viz/viz8_confusion_matrix_comparison.png", "Confusion Matrix Comparison"),
        ("viz/viz5_perclass_performance_baseline.png", "Per-Class Performance"),
        ("viz/viz9_violation_reduction.png", "Clinical Rule Violation Reduction"),
        ("viz/viz15_ablation_study.png", "Ablation Study")
    ]

    for idx, (viz_file, title) in enumerate(viz_files):
        try:
            with viz_cols[idx % 2]:
                st.image(viz_file, caption=title, use_container_width=True)
        except:
            pass

    # Trajectory model performance
    st.divider()
    st.subheader("📉 Trajectory Model Performance")

    try:
        st.image("viz/evaluation_metrics.png", caption="Neural ODE Trajectory Prediction Performance", 
                use_container_width=True)
    except:
        st.info("Trajectory model visualizations not available")

def about_page(metadata):
    """About page"""
    st.header("ℹ️ About DiaSight")
    st.markdown("Learn more about DiaSight's AI-powered healthcare solutions.")

    st.divider()

    # Logo placeholder
    logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
    with logo_col2:
        # Try to load logo if available, otherwise show placeholder
        try:
            st.image("assets/DiaSight_Logo_Vector.svg", use_container_width=True)
        except:
            st.info("📌 Logo Placeholder (1920 x 589 px) - Upload DiaSight_Logo_Vector.svg")    

    # Title
    st.markdown("<h2 style='text-align: center;'>DiaSight: Pioneering Accessible Diabetic Retinopathy Screening</h2>", unsafe_allow_html=True)

    # Create tabs
    tab1 = st.tabs([
        "🚀 Our Journey"
    ])[0]

    with tab1:
        st.header("DiaSight: A Journey of Innovation and Impact")

        st.markdown("### National AI Fest 2025 | August 11-13, Iloilo Convention Center")
        st.markdown("🥈 **1st Runner-Up**")
        # SHOW 4 IMAGES HERE: 1_AIFEST, 2_AIFEST, 3_AIFEST, 11_AIFEST
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            try:
                st.image("assets/1_AIFEST.jpg", use_container_width=True)
            except:
                st.info("Image: 1_AIFEST.jpg")
        with col2:
            try:
                st.image("assets/2_AIFEST.jpg", use_container_width=True)
            except:
                st.info("Image: 2_AIFEST.jpg")
        with col3:
            try:
                st.image("assets/3_AIFEST.jpg", use_container_width=True)
            except:
                st.info("Image: 3_AIFEST.jpg")
        with col4:
            try:
                st.image("assets/11_AIFEST.jpg", use_container_width=True)
            except:
                st.info("Image: 11_AIFEST.jpg")

        st.markdown("""
        Our journey began at the inaugural National AI Fest, organized by DOST Region VI with support from AWS, PCIEERD, PCHRD, and the Analytics Association of the Philippines. Under the theme "Coding a Better Future: Responsible AI for Cities and Communities," we competed against the nation's brightest innovators and earned 1st Runner-Up recognition.

        The festival wasn't just about competition—it opened doors to invaluable mentorship. We spent three intensive days learning from Dr. Romulo de Castro, a renowned bioinformatician and genomics expert specializing in AI for cancer research, and Mr. Andres Montiel of Packworks. Their guidance shaped how we think about AI's real-world impact in healthcare.
        """)

        st.markdown("---")

        st.markdown("### AI.DEAS for Impact 2025 | September 9-10, Region 6")
        st.markdown("🏆 **Champion** | ⭐ **Special Award for Visionary Innovators**")
        # SHOW IMAGE HERE: 4_AIDEAS
        try:
            st.image("assets/4_AIDEAS.jpg", use_container_width=True)
        except:
            st.info("Image: 4_AIDEAS.jpg")

        st.markdown("""
        Just weeks later, we took home the championship at DICT's AI.DEAS for Impact hackathon in Bacolod. This regional competition, co-organized with the Analytics Association of the Philippines, challenged teams to develop ethical, applicable AI solutions for real societal problems.

        We didn't just win—we earned the Special Award for Visionary Innovators, recognizing our potential to transform diabetic retinopathy screening in the Philippines. This double victory validated our mission: making advanced healthcare accessible through AI, especially for underserved communities.
        """)

        st.markdown("---")

        st.markdown("### PSC X Regional Pitching Competition | October 24, Western Visayas")
        st.markdown("🏆 **Champion** | 🎤 **Best Pitch** | 💡 **Most Innovative Concept**")
        # SHOW IMAGE HERE: 5_PSCX
        try:
            st.image("assets/5_PSCX.jpg", use_container_width=True)
        except:
            st.info("Image: 5_PSCX.jpg")

        st.markdown("""
        At DICT's Philippine Startup Challenge X Regional Pitching, we swept the competition with a triple crown: Champion, Best Pitch, and Innovative Awardee. Competing against startups across Western Visayas, we demonstrated not just technical excellence but also business viability and market readiness.

        This clean sweep earned us a spot to represent Region VI at the national finals—a crucial step toward scaling DiaSight nationwide.
        """)

        st.markdown("---")

        st.markdown("### WESTnovation Challenge 2025 | November 14, WVSU Iloilo")
        st.markdown("🏆 **Champion** | 🚀 **Most Market-Ready Innovation**")
        # SHOW IMAGE HERE: 6_WESTNOV, 7_WESTNOV
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.image("assets/6_WESTNOV.jpg", use_container_width=True)
            except:
                st.info("Image: 6_WESTNOV.jpg")
        with col2:
            try:
                st.image("assets/7_WESTNOV.jpg", use_container_width=True)
            except:
                st.info("Image: 7_WESNOV.jpg")

        st.markdown("""
        Organized by DOST R6, DEPDev R6, and DevCon Iloilo, the WESTnovation Challenge tested our innovation's commercial potential. We proved that DiaSight wasn't just a research project—it's a market-ready solution ready to deploy.

        Winning "Most Market-Ready Innovation" alongside the championship title showed that we've bridged the gap between academic research and real-world implementation. Our AI tool can work today, in clinics today, saving vision today.
        """)

        st.markdown("---")

        st.markdown("### Innovation in Action: BINHI-TBI Partnership")
        st.markdown("📝 **Contract Signing**")
        # Show image here: 8_INCUBATE
        try:
            st.image("assets/8_INCUBATE.jpg", use_container_width=True)
        except:
            st.info("Image: 8_INCUBATE.jpg")

        st.markdown("""
        Recognition turned into reality when we signed with West Visayas State University's BINHI Technology Business Incubator in partnership with DOST. As a student-led startup, this marked our transition from competition winners to officially incubated innovators with institutional backing to scale our impact.
        """)

        st.markdown("---")

        st.markdown("### 26th International Symposium on Advanced Intelligent Systems | November 6-9, Cheongju, South Korea")
        st.markdown("📄 **Published Research (Top 5%)**")
        # Show images here: 9_KOREA, 10_KOREA
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.image("assets/9_KOREA.jpg", use_container_width=True)
            except:
                st.info("Image: 9_KOREA.jpg")
        with col2:
            try:
                st.image("assets/10_KOREA.jpg", use_container_width=True)
            except:
                st.info("Image: 10_KOREA.jpg")

        st.markdown("""
        Our journey reached the international stage at the 26th International Symposium on Advanced Intelligent Systems at Chungbuk National University. Our paper, "Non-Invasive Diabetic Retinopathy Risk Stratification with XAI-Enabled Ensemble Machine Learning on Augmented EHRs," placed in the top 5% of submissions.

        Co-organized by KIIS (Korean Institute of Intelligent Systems), IEEE, the Japan Society for Fuzzy Theory and Intelligent Informatics, and the Taiwan Fuzzy Systems Association, ISIS 2025 brought together the world's leading minds in AI and intelligent systems. Presenting alongside established researchers validated our approach: combining explainable AI with ensemble learning to make DR screening accessible without expensive imaging equipment. Presenting to medical professionals, biomedical engineers, and experts in the field of AI and computing.
        """)

        st.markdown("---")

        st.markdown("### Philippine Startup Challenge X National Finals | December 3-4, 2025")
        st.markdown("🏆 **National Champion** | 💥 **Most Disruptive Idea** | 📈 **Best Business Model**")
        # SHOW 3 IMAGES HERE: 12_PSCX, 13_PSCX, 14_PSCX
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                st.image("assets/12_PSCX.jpg", use_container_width=True)
            except:
                st.info("Image: 12_PSCX.jpg")
        with col2:
            try:
                st.image("assets/13_PSCX.jpg", use_container_width=True)
            except:
                st.info("Image: 13_PSCX.jpg")
        with col3:
            try:
                st.image("assets/14_PSCX.jpg", use_container_width=True)
            except:
                st.info("Image: 14_PSCX.jpg")

        st.markdown("""
        DiaSight is grateful to have been named National Champion at the Philippine Startup Challenge X in 2025, where our work on lab-based diabetic retinopathy screening was also recognized with the awards for Most Disruptive Idea and Best Business Model.
        """)

    # Footer
    st.markdown("---")
    st.caption("DiaSight - Pioneering Accessible Diabetic Retinopathy Screening")
    st.caption("⚠️ For educational purposes only. Not a substitute for professional medical judgment.")

if __name__ == "__main__":
    main()