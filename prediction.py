"""
prediction.py
--------------
Loads the trained model pipeline and exposes a simple function to
predict a student's final score from raw input values.
"""

import joblib
import pandas as pd

MODEL_PATH = "models/student_model.pkl"

_model = None  # lazy-loaded singleton


def load_model(path: str = MODEL_PATH):
    """Load (and cache) the trained model pipeline."""
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


def predict_performance(
    hours_studied: float,
    attendance: float,
    previous_scores: float,
    sleep_hours: float,
    study_hours_per_week: float,
    extracurricular: str,
    internet_access: str,
    parental_education: str,
) -> float:
    """
    Predict a student's Final_Score given their attributes.

    Returns a float score clipped to the 0-100 range.
    """
    model = load_model()

    input_df = pd.DataFrame(
        [
            {
                "Hours_Studied": hours_studied,
                "Attendance": attendance,
                "Previous_Scores": previous_scores,
                "Sleep_Hours": sleep_hours,
                "Study_Hours_Per_Week": study_hours_per_week,
                "Extracurricular_Activities": extracurricular,
                "Internet_Access": internet_access,
                "Parental_Education": parental_education,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    return round(float(min(max(prediction, 0), 100)), 2)


def score_to_grade(score: float) -> str:
    """Convert a numeric score to a letter grade."""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


if __name__ == "__main__":
    # Quick manual test
    result = predict_performance(
        hours_studied=6,
        attendance=85,
        previous_scores=70,
        sleep_hours=7,
        study_hours_per_week=40,
        extracurricular="Yes",
        internet_access="Yes",
        parental_education="Graduate",
    )
    print(f"Predicted Final Score: {result} (Grade: {score_to_grade(result)})")
