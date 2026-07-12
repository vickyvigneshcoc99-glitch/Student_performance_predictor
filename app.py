"""
app.py
------
Streamlit dashboard for the Student Performance Predictor.

Run:
    streamlit run app.py
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from prediction import predict_performance, score_to_grade

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "student_model.pkl"


st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

st.set_option("client.showErrorDetails", False)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def get_summary_stats(df: pd.DataFrame):
    return {
        "shape": df.shape,
        "head": df.head(10).to_string(index=False),
        "describe": df.describe().to_string(),
        "corr": df.select_dtypes(include="number").corr()["Final_Score"].sort_values(ascending=False).to_string(),
    }


def main():
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
        summary = get_summary_stats(df)

        st.write(f"**Shape:** {summary['shape'][0]} rows × {summary['shape'][1]} columns")
        st.write("Sample rows:")
        st.write(summary["head"])

        st.subheader("Summary Statistics")
        st.write(summary["describe"])

        st.subheader("Final Score Distribution")
        st.write("The dataset contains student scores ranging from low to high performance levels.")

        st.subheader("Correlation Summary")
        st.write(summary["corr"])

    # ---------------- Prediction ----------------
    elif page == "Predict Performance":
        st.title("🔮 Predict Student Performance")

        if not os.path.exists(MODEL_PATH):
            st.error("Model not found. Please run `python train_model.py` first.")
        else:
            st.write("Enter the student's details below:")
            hours_studied = st.text_input("Hours Studied (per day)", value="5")
            attendance = st.text_input("Attendance (%)", value="80")
            previous_scores = st.text_input("Previous Scores", value="65")
            sleep_hours = st.text_input("Sleep Hours (per day)", value="7")
            study_hours_per_week = st.text_input("Study Hours Per Week", value="35")
            extracurricular = st.text_input("Extracurricular Activities (Yes/No)", value="Yes")
            internet_access = st.text_input("Internet Access (Yes/No)", value="Yes")
            parental_education = st.text_input(
                "Parental Education (High School/Graduate/Postgraduate)",
                value="Graduate",
            )

            st.markdown("---")

            if st.button("Predict Performance 🚀", type="primary"):
                try:
                    score = predict_performance(
                        hours_studied=float(hours_studied),
                        attendance=float(attendance),
                        previous_scores=float(previous_scores),
                        sleep_hours=float(sleep_hours),
                        study_hours_per_week=float(study_hours_per_week),
                        extracurricular=extracurricular.strip().title(),
                        internet_access=internet_access.strip().title(),
                        parental_education=parental_education.strip().title(),
                    )
                    grade = score_to_grade(score)

                    st.metric("Predicted Final Score", f"{score} / 100")
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
                except ValueError:
                    st.error("Please enter valid numeric values for the score fields.")


if __name__ == "__main__":
    main()

app = main
application = app
