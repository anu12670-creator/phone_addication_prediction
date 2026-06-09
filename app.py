import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Behavioral Analytics | Phone Addiction Predictor", 
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling for Professional Look ---
st.markdown("""
    <style>
    .main-header {
        font-size:2.5rem !important;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size:1.1rem !important;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .card-container {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E3A8A;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data & Model Loading (Cached with Reliable Paths) ---
@st.cache_resource
def load_model():
    # Direct path where project.py creates the pkl file
    model_path = r"C:\Users\dell\Desktop\ML_Project\GradientBoosting_final_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_clean_schema():
    # Direct path to your data source
    csv_path = r"C:\Users\dell\Desktop\ML_Project\phone_addiction_dataset.csv"
    try:
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            target_col = "Addiction_Level"
            data = data.drop_duplicates().dropna(subset=[target_col])
            columns_to_drop = ["Name", "Location"] 
            data = data.drop(columns=columns_to_drop, errors="ignore")
            return data.drop(columns=[target_col])
        else:
            return None
    except Exception as e:
        st.error(f"Error loading dataset schema: {e}")
        return None

# Initialize data structures
pipeline = load_model()
X_schema = load_clean_schema()

# --- Header Architecture ---
st.markdown('<div class="main-header">📱 Digital Wellbeing & Behavioral Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predictive Modeling Engine for Assessment of Smartphone Dependency Metrics</div>', unsafe_allow_html=True)

if X_schema is None:
    st.error("⚠️ **Schema Map Offline:** Unable to locate dataset file at 'C:\\Users\\dell\\Desktop\\ML_Project\\phone_addiction_dataset.csv'.")
elif pipeline is None:
    st.error("⚠️ **Model Engine Offline:** 'GradientBoosting_final_model.pkl' could not be found. Please ensure project.py has run and generated this file in your folder.")
else:
    # Segregate metrics
    cat_cols = X_schema.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_schema.select_dtypes(include=['number']).columns.tolist()

    # Split features into 2 balanced columns layout for side-by-side presentation
    col_left, col_right = st.columns(2, gap="large")
    input_data = {}

    # Halfway point to split numerical inputs evenly between left and right columns
    half_num = len(num_cols) // 2

    # --- LEFT COLUMN: Core Usage Metrics & Categorical Data ---
    with col_left:
        st.markdown("### 📊 Screen Time & App Usage")
        st.caption("Quantitative data regarding screen interaction intervals")
        
        # First half of numerical features
        for col in num_cols[:half_num]:
            min_val = float(X_schema[col].min())
            max_val = float(X_schema[col].max())
            median_val = float(X_schema[col].median())
            step = 1.0 if (max_val - min_val) > 10 else 0.1
            
            input_data[col] = st.number_input(
                label=f"🔢 {col.replace('_', ' ')}",
                min_value=min_val, 
                max_value=max_val, 
                value=median_val, 
                step=step,
                key=f"left_{col}"
            )
            
        st.markdown("---")
        st.markdown("### 🔠 Categorical Profiles")
        st.caption("Qualitative behavioral demographics")
        
        # All categorical features placed under the first column cluster
        for col in cat_cols:
            unique_options = X_schema[col].dropna().unique().tolist()
            if not unique_options: 
                unique_options = ['Unknown']
            
            input_data[col] = st.selectbox(
                label=f"🔹 {col.replace('_', ' ')}",
                options=unique_options,
                key=f"cat_{col}"
            )

    # --- RIGHT COLUMN: Psychological & Lifestyle Elements ---
    with col_right:
        st.markdown("### 🧠 Psychological & Lifestyle Metrics")
        st.caption("Self-reported health, anxiety, and performance parameters")
        
        # Remaining half of the numerical features
        for col in num_cols[half_num:]:
            min_val = float(X_schema[col].min())
            max_val = float(X_schema[col].max())
            median_val = float(X_schema[col].median())
            step = 1.0 if (max_val - min_val) > 10 else 0.1
            
            input_data[col] = st.number_input(
                label=f"📈 {col.replace('_', ' ')}",
                min_value=min_val, 
                max_value=max_val, 
                value=median_val, 
                step=step,
                key=f"right_{col}"
            )

    # --- PREDICTION RUNTIME ---
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📊 Evaluate Diagnostic Metrics", type="primary", use_container_width=True):
        
        # Parse and match column orientation perfectly with train sequence
        input_df = pd.DataFrame([input_data])[X_schema.columns]
        
        try:
            # Generate raw prediction value
            raw_prediction = float(pipeline.predict(input_df)[0])
            
            # Constrain prediction bounds perfectly between 0.00 and 10.00
            prediction = np.clip(raw_prediction, 0.0, 10.0)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.subheader("📋 Assessment Diagnostic Summary")
            
            # Setup interactive layout for results dashboard
            res_col1, res_col2 = st.columns([1, 2], gap="medium")
            
            with res_col1:
                st.metric(label="Calculated Addiction Index", value=f"{prediction:.2f}")
                
                # Normalize the visual bar representation safely
                normalized_val = prediction / 10.0 
                st.progress(normalized_val)
                
            with res_col2:
                if prediction < 3.5:
                    st.success("### ✅ Status: Low/Normal Affinity\nPatient displays balanced structural behavior patterns with digital utilities. No immediate interventions required.")
                elif 3.5 <= prediction < 7.0:
                    st.warning("### ⚠️ Status: Moderate Habituation Risk\nElevated interactions recorded. Marginal impact discovered across psychological or sleep dimensions. Recommend boundary schedules.")
                else:
                    st.error("### 🚨 Status: High Critical Dependency\nSevere reliance indicators detected. Critical correlations found with anxiety, sleep deprivation, or operational performance metrics. Immediate digital detox protocols recommended.")
                    
        except Exception as e:
            st.error(f"Critical Runtime Exception mapping feature structures: {e}")
