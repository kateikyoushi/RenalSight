# RenalSight: AI-Powered Chronic Kidney Disease Risk Assessment System

## Overview

RenalSight is a comprehensive AI-powered clinical decision support system developed by DiaSight for Chronic Kidney Disease (CKD) assessment in diabetic patients. The system provides two complementary analyses: current CKD stage prediction and disease progression trajectory forecasting, designed to assist healthcare professionals in making informed clinical decisions.

## Features

### 🏥 Patient Risk Assessment
- **Stage Assessment**: Predicts current CKD stage (1-5) using Neuro-Symbolic AI that combines machine learning with clinical rule integration
- **Trajectory Prediction**: Forecasts disease progression using Neural ODE technology to predict stable vs. progressive trajectories
- **Clinical Rule Integration**: Incorporates KDIGO guidelines and clinical best practices
- **Explainable AI**: SHAP-based feature importance analysis for transparency

### 🏥 Referral Directory
- Comprehensive YAKAP clinics database with 2300+ verified facilities
- Advanced search and filtering capabilities
- Direct email referral generation
- Pagination for optimal performance

### 📊 Model Performance Analytics
- Real-time model performance metrics
- Comparative analysis of different AI approaches
- Visual performance dashboards

## Technical Architecture

### Stage Prediction Model
- **Algorithm**: XGBoost classifier with Optuna optimization
- **Symbolic Component**: Clinical rules for BUN, UACR, and Albumin thresholds
- **Features**: 11 clinical parameters (age, sex, diabetes duration, vitals, lab values)
- **Approach**: Soft rule guidance for improved accuracy

### Trajectory Prediction Model
- **Algorithm**: Neural ODE Classifier
- **ODE Solver**: DOPRI5 adaptive integration
- **Features**: 12 parameters including current eGFR
- **Output**: Binary classification (Stable vs. Progressive trajectory)

## Clinical Applications

### Stage Assessment Answers:
- What is the patient's current CKD stage?
- Are there any clinical rule violations?
- Which biomarkers are concerning?

### Trajectory Prediction Answers:
- Will the patient's kidney function decline rapidly?
- When might the patient need dialysis?
- What is the expected rate of eGFR decline?
- Is urgent intervention needed?

## About DiaSight

DiaSight is a pioneering healthcare AI company focused on making advanced medical screening accessible through innovative technology. Founded as a student-led startup, DiaSight has rapidly gained recognition in the Philippine startup ecosystem:

- 🥈 **1st Runner-Up** - National AI Fest 2025
- 🏆 **Champion** - AI.DEAS for Impact 2025
- 🏆 **Triple Crown Winner** - PSC X Regional Pitching Competition
- 🏆 **National Champion** - Philippine Startup Challenge X 2025
- 📄 **Top 5% Research Publication** - International Symposium on Advanced Intelligent Systems

DiaSight specializes in diabetic retinopathy screening and has expanded into comprehensive diabetic care solutions, including RenalSight for CKD risk assessment.

## Clinical Disclaimer

**Important**: This tool is designed to **assist** healthcare professionals and should **NOT** replace clinical judgment. All predictions should be validated with comprehensive patient assessment and appropriate diagnostic tests.

- Always correlate with clinical presentation
- Consider additional diagnostic tests
- Follow local guidelines and protocols
- Document appropriately in medical records

## Dataset Information

- **Sample Size**: 503 diabetic patients with CKD stages 1-5
- **Validation**: 60/20/20 train/validation/test split
- **Class Balancing**: SMOTE-ENC technique
- **Performance**: Test accuracy ~95% for stage prediction, 97.8% for trajectory prediction

## Contact

For more information about DiaSight and our healthcare AI solutions, visit our website or contact us through official channels.

---

*RenalSight v2.0 - Integrated Stage & Trajectory Assessment by DiaSight*
*For educational and research purposes only*