"""
Heart Disease Prediction Web Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
    with open('logistic_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('decision_tree_model.pkl', 'rb') as f:
        dt_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return lr_model, dt_model, scaler, feature_names

@st.cache_data
def load_data():
    df = pd.read_csv('heart_disease_clean.csv')
    return df

lr_model, dt_model, scaler, feature_names = load_models()
df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Exploration", "🔮 Prediction", "📈 Model Performance"])

# Home Page
if page == "🏠 Home":
    st.title("❤️ Heart Disease Prediction System")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Best Model Accuracy", "83.3%")

    st.markdown("---")
    st.header("About This Project")
    st.write("""
    This application predicts the risk of heart disease using machine learning models trained on the UCI Heart Disease dataset.

    **Key Features:**
    - Interactive data exploration and visualization
    - Real-time heart disease risk prediction
    - Model performance comparison
    - Feature importance analysis

    **Models Used:**
    - Logistic Regression (ROC AUC: 0.95)
    - Decision Tree (ROC AUC: 0.75)
    """)

    st.markdown("---")
    st.header("Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.header("Feature Descriptions")
    feature_desc = {
        "age": "Age in years",
        "sex": "Sex (1 = male, 0 = female)",
        "cp": "Chest pain type (1-4)",
        "trestbps": "Resting blood pressure (mm Hg)",
        "chol": "Serum cholesterol (mg/dl)",
        "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
        "restecg": "Resting electrocardiographic results (0-2)",
        "thalach": "Maximum heart rate achieved",
        "exang": "Exercise induced angina (1 = yes, 0 = no)",
        "oldpeak": "ST depression induced by exercise",
        "slope": "Slope of peak exercise ST segment (1-3)",
        "ca": "Number of major vessels colored by fluoroscopy (0-3)",
        "thal": "Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)"
    }

    for feature, description in feature_desc.items():
        st.write(f"**{feature}**: {description}")

# Data Exploration Page
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Correlation Analysis", "Statistical Summary"])

    with tab1:
        st.header("Feature Distributions")

        col1, col2 = st.columns(2)

        with col1:
            # Target distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            df['target'].value_counts().plot(kind='bar', color=['green', 'red'], ax=ax)
            ax.set_title('Target Distribution', fontweight='bold')
            ax.set_xlabel('Target (0=No Disease, 1=Disease)')
            ax.set_ylabel('Count')
            ax.set_xticklabels(['No Disease', 'Disease'], rotation=0)
            st.pyplot(fig)

            # Age distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist([df[df['target']==0]['age'], df[df['target']==1]['age']],
                   bins=15, label=['No Disease', 'Disease'], color=['green', 'red'], alpha=0.7)
            ax.set_title('Age Distribution by Target', fontweight='bold')
            ax.set_xlabel('Age')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)

        with col2:
            # Sex distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            sex_target = pd.crosstab(df['sex'], df['target'])
            sex_target.plot(kind='bar', ax=ax, color=['green', 'red'])
            ax.set_title('Sex vs Target', fontweight='bold')
            ax.set_xlabel('Sex (0=Female, 1=Male)')
            ax.set_ylabel('Count')
            ax.set_xticklabels(['Female', 'Male'], rotation=0)
            ax.legend(['No Disease', 'Disease'])
            st.pyplot(fig)

            # Chest pain type
            fig, ax = plt.subplots(figsize=(8, 5))
            cp_target = pd.crosstab(df['cp'], df['target'])
            cp_target.plot(kind='bar', ax=ax, color=['green', 'red'])
            ax.set_title('Chest Pain Type vs Target', fontweight='bold')
            ax.set_xlabel('Chest Pain Type')
            ax.set_ylabel('Count')
            ax.legend(['No Disease', 'Disease'])
            st.pyplot(fig)

    with tab2:
        st.header("Correlation Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Correlation heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            corr_matrix = df.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation Heatmap', fontweight='bold')
            st.pyplot(fig)

        with col2:
            # Correlation with target
            st.subheader("Correlation with Target")
            target_corr = df.corr()['target'].sort_values(ascending=False)
            st.dataframe(target_corr.to_frame('Correlation'), use_container_width=True)

    with tab3:
        st.header("Statistical Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("No Disease (0)")
            st.dataframe(df[df['target']==0].describe(), use_container_width=True)

        with col2:
            st.subheader("Disease (1)")
            st.dataframe(df[df['target']==1].describe(), use_container_width=True)

# Prediction Page
elif page == "🔮 Prediction":
    st.title("🔮 Heart Disease Risk Prediction")
    st.markdown("---")

    st.write("Enter patient information to predict heart disease risk:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col3:
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", options=[1, 2, 3])
        ca = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", options=[3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])

    st.markdown("---")

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
            'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
            'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
        })

        # Scale for Logistic Regression
        input_scaled = scaler.transform(input_data)

        # Predictions
        lr_pred = lr_model.predict(input_scaled)[0]
        lr_proba = lr_model.predict_proba(input_scaled)[0]

        dt_pred = dt_model.predict(input_data)[0]
        dt_proba = dt_model.predict_proba(input_data)[0]

        st.markdown("---")
        st.header("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Logistic Regression")
            if lr_pred == 1:
                st.error("⚠️ High Risk of Heart Disease")
            else:
                st.success("✅ Low Risk of Heart Disease")

            st.metric("Disease Probability", f"{lr_proba[1]*100:.1f}%")
            st.progress(lr_proba[1])

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(['No Disease', 'Disease'], lr_proba, color=['green', 'red'])
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities', fontweight='bold')
            ax.set_ylim([0, 1])
            for i, v in enumerate(lr_proba):
                ax.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontweight='bold')
            st.pyplot(fig)

        with col2:
            st.subheader("Decision Tree")
            if dt_pred == 1:
                st.error("⚠️ High Risk of Heart Disease")
            else:
                st.success("✅ Low Risk of Heart Disease")

            st.metric("Disease Probability", f"{dt_proba[1]*100:.1f}%")
            st.progress(dt_proba[1])

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(['No Disease', 'Disease'], dt_proba, color=['green', 'red'])
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities', fontweight='bold')
            ax.set_ylim([0, 1])
            for i, v in enumerate(dt_proba):
                ax.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontweight='bold')
            st.pyplot(fig)

        st.markdown("---")
        st.info("**Note:** This is a predictive model for educational purposes. Always consult healthcare professionals for medical advice.")

# Model Performance Page
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Analysis")
    st.markdown("---")

    # Load test data for evaluation
    X = df.drop('target', axis=1)
    y = df['target']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)

    # Get predictions
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    dt_pred = dt_model.predict(X_test)
    dt_proba = dt_model.predict_proba(X_test)[:, 1]

    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Confusion Matrices", "Feature Importance"])

    with tab1:
        st.header("Model Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Accuracy Scores")
            from sklearn.metrics import accuracy_score
            lr_acc = accuracy_score(y_test, lr_pred)
            dt_acc = accuracy_score(y_test, dt_pred)

            fig, ax = plt.subplots(figsize=(8, 5))
            models = ['Logistic\nRegression', 'Decision\nTree']
            accuracies = [lr_acc, dt_acc]
            bars = ax.bar(models, accuracies, color=['steelblue', 'forestgreen'])
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy Comparison', fontweight='bold')
            ax.set_ylim([0, 1])
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{acc:.3f}', ha='center', fontweight='bold')
            st.pyplot(fig)

        with col2:
            st.subheader("ROC Curves")
            from sklearn.metrics import roc_curve, auc

            lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
            dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_proba)
            lr_auc = auc(lr_fpr, lr_tpr)
            dt_auc = auc(dt_fpr, dt_tpr)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC={lr_auc:.3f})', linewidth=2)
            ax.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC={dt_auc:.3f})', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

    with tab2:
        st.header("Confusion Matrices")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Logistic Regression")
            cm_lr = confusion_matrix(y_test, lr_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title('Confusion Matrix', fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            st.pyplot(fig)

            from sklearn.metrics import classification_report
            st.text("Classification Report:")
            st.text(classification_report(y_test, lr_pred))

        with col2:
            st.subheader("Decision Tree")
            cm_dt = confusion_matrix(y_test, dt_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax)
            ax.set_title('Confusion Matrix', fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            st.pyplot(fig)

            st.text("Classification Report:")
            st.text(classification_report(y_test, dt_pred))

    with tab3:
        st.header("Feature Importance")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Logistic Regression")
            lr_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': np.abs(lr_model.coef_[0])
            }).sort_values('Coefficient', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(range(len(lr_importance)), lr_importance['Coefficient'], color='steelblue')
            ax.set_yticks(range(len(lr_importance)))
            ax.set_yticklabels(lr_importance['Feature'])
            ax.set_xlabel('Absolute Coefficient')
            ax.set_title('Feature Importance', fontweight='bold')
            ax.invert_yaxis()
            st.pyplot(fig)

            st.dataframe(lr_importance, use_container_width=True)

        with col2:
            st.subheader("Decision Tree")
            dt_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': dt_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(range(len(dt_importance)), dt_importance['Importance'], color='forestgreen')
            ax.set_yticks(range(len(dt_importance)))
            ax.set_yticklabels(dt_importance['Feature'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance', fontweight='bold')
            ax.invert_yaxis()
            st.pyplot(fig)

            st.dataframe(dt_importance, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("**Heart Disease Prediction System**\n\nBuilt with Streamlit and scikit-learn")
