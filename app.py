"""
app.py
------
Streamlit dashboard for the Student Performance Predictor.

Run:
    streamlit run app.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from prediction import predict_performance, score_to_grade

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

DATA_PATH = "student_dataset.csv"
MODEL_PATH = "models/student_model.pkl"


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


# ---------------- Sidebar ----------------
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset Overview", "Predict Performance"])

st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard predicts a student's final score based on study habits, "
    "attendance, and other factors using a trained Machine Learning model."
)

# ---------------- Home ----------------
if page == "Home":
    st.title("🎓 Student Performance Predictor")
    st.markdown(
        """
        Welcome! This mini-project uses a **Random Forest Regression** model
        to predict a student's final academic score based on:

        - Hours studied per day
        - Class attendance
        - Previous exam scores
        - Sleep hours
        - Weekly study hours
        - Extracurricular activity involvement
        - Internet access
        - Parental education level

        Use the sidebar to explore the dataset or make a live prediction.
        """
    )
    if os.path.exists(MODEL_PATH):
        st.success("✅ Trained model found and ready to use.")
    else:
        st.warning("⚠️ Model not found. Run `python train_model.py` first.")

# ---------------- Dataset Overview ----------------
elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    df = load_dataset()

    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    with col2:
        st.subheader("Final Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Final_Score"], kde=True, ax=ax, color="#4C72B0")
        ax.set_xlabel("Final Score")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include="number")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ---------------- Prediction ----------------
elif page == "Predict Performance":
    st.title("🔮 Predict Student Performance")

    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please run `python train_model.py` first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            hours_studied = st.slider("Hours Studied (per day)", 0.0, 12.0, 5.0, 0.5)
            attendance = st.slider("Attendance (%)", 40.0, 100.0, 80.0, 1.0)
            previous_scores = st.slider("Previous Scores", 0.0, 100.0, 65.0, 1.0)
            sleep_hours = st.slider("Sleep Hours (per day)", 3.0, 10.0, 7.0, 0.5)

        with col2:
            study_hours_per_week = st.slider(
                "Study Hours Per Week", 0.0, 80.0, 35.0, 1.0
            )
            extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
            internet_access = st.selectbox("Internet Access", ["Yes", "No"])
            parental_education = st.selectbox(
                "Parental Education", ["High School", "Graduate", "Postgraduate"]
            )

        st.markdown("---")

        if st.button("Predict Performance 🚀", type="primary"):
            score = predict_performance(
                hours_studied=hours_studied,
                attendance=attendance,
                previous_scores=previous_scores,
                sleep_hours=sleep_hours,
                study_hours_per_week=study_hours_per_week,
                extracurricular=extracurricular,
                internet_access=internet_access,
                parental_education=parental_education,
            )
            grade = score_to_grade(score)

            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.metric("Predicted Final Score", f"{score} / 100")
            with result_col2:
                st.metric("Predicted Grade", grade)

            st.progress(min(int(score), 100))

            if score >= 70:
                st.success("Great! The student is predicted to perform well. 🎉")
            elif score >= 50:
                st.info("The student is predicted to pass with a moderate score.")
            else:
                st.warning(
                    "The student is at risk of low performance. "
                    "Consider more study hours and better attendance."
                )
