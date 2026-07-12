"""
generate_dataset.py
--------------------
Generates a realistic synthetic student performance dataset and
saves it as student_dataset.csv. Run this once before train_model.py
(a dataset is already included, so this is optional / for reference).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000

hours_studied = np.random.normal(5, 2, N).clip(0, 12)
attendance = np.random.normal(80, 12, N).clip(40, 100)
previous_scores = np.random.normal(65, 15, N).clip(30, 100)
sleep_hours = np.random.normal(6.5, 1.5, N).clip(3, 10)
extracurricular = np.random.choice(["Yes", "No"], N, p=[0.4, 0.6])
internet_access = np.random.choice(["Yes", "No"], N, p=[0.75, 0.25])
parental_education = np.random.choice(
    ["High School", "Graduate", "Postgraduate"], N, p=[0.4, 0.4, 0.2]
)
study_hours_per_week = hours_studied * 7 + np.random.normal(0, 2, N)

# Underlying relationship used to generate a realistic target score
base_score = (
    0.35 * previous_scores
    + 2.2 * hours_studied
    + 0.25 * attendance
    + 1.0 * sleep_hours
    + (extracurricular == "Yes") * 2
    + (internet_access == "Yes") * 3
    + pd.Series(parental_education).map(
        {"High School": 0, "Graduate": 3, "Postgraduate": 6}
    ).values
    + np.random.normal(0, 5, N)
)

final_score = np.clip(base_score * 0.7, 0, 100).round(2)

df = pd.DataFrame(
    {
        "Hours_Studied": hours_studied.round(2),
        "Attendance": attendance.round(2),
        "Previous_Scores": previous_scores.round(2),
        "Sleep_Hours": sleep_hours.round(2),
        "Extracurricular_Activities": extracurricular,
        "Internet_Access": internet_access,
        "Parental_Education": parental_education,
        "Study_Hours_Per_Week": study_hours_per_week.round(2),
        "Final_Score": final_score,
    }
)

df.to_csv("student_dataset.csv", index=False)
print(f"Dataset generated: student_dataset.csv ({len(df)} rows)")
print(df.head())
