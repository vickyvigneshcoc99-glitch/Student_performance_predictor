# 🎓 Student Performance Predictor

A machine learning mini-project that predicts a student's final academic
score based on study habits, attendance, sleep, and other lifestyle
factors — with an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
Student_Performance_Predictor/
│
├── app.py                  # Streamlit dashboard (main entry point)
├── train_model.py          # Trains the ML model and saves it
├── prediction.py           # Prediction logic used by the dashboard
├── generate_dataset.py     # (Optional) regenerates the synthetic dataset
├── student_dataset.csv     # Dataset used for training
├── requirements.txt        # Python dependencies
├── README.md                # This file
│
├── models/
│   └── student_model.pkl    # Saved trained ML pipeline
│
└── notebooks/
    └── Student_Performance.ipynb   # (Optional) EDA / experimentation notebook
```

---

## ⚙️ Setup Instructions (VS Code)

1. **Open the folder in VS Code**
   ```
   File → Open Folder → Student_Performance_Predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (already trained, but you can retrain anytime)
   ```bash
   python train_model.py
   ```
   This reads `student_dataset.csv`, trains a `RandomForestRegressor`,
   prints evaluation metrics (MAE, RMSE, R²), and saves the model to
   `models/student_model.pkl`.

5. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```
   This opens the app in your browser at `http://localhost:8501`.

---

## 🧠 How It Works

- **Features used:** Hours Studied, Attendance, Previous Scores, Sleep
  Hours, Weekly Study Hours, Extracurricular Activities, Internet Access,
  Parental Education.
- **Target:** `Final_Score` (0–100).
- **Model:** `RandomForestRegressor` inside a scikit-learn `Pipeline`
  that also handles scaling (numeric features) and one-hot encoding
  (categorical features), so `prediction.py` can pass raw values
  straight in.
- **Dashboard pages:**
  - **Home** – project overview and model status.
  - **Dataset Overview** – summary stats, score distribution, and a
    correlation heatmap.
  - **Predict Performance** – sliders/dropdowns to enter a student's
    details and get a live predicted score + grade.

---

## 🔁 Regenerating the Dataset

The included `student_dataset.csv` is synthetic but statistically
realistic (1000 rows). To regenerate it with a different sample size or
random seed, edit and run:

```bash
python generate_dataset.py
```

---

## 📸 Screenshots

Add your own screenshots of the running dashboard to an `images/` or
`screenshots/` folder (e.g. `dashboard.png`, `prediction.png`) after
running `streamlit run app.py` — these are great for your internship
report/PPT.

---

## 🛠️ Tech Stack

- Python 3.9+
- scikit-learn (ML model)
- Streamlit (dashboard)
- pandas / numpy (data handling)
- matplotlib / seaborn (visualization)
- joblib (model persistence)

---

## � Deployment

This project is configured to deploy as a Streamlit web app.

### Recommended host: Render

1. Go to https://dashboard.render.com/
2. Create a free account or log in.
3. Click **New** → **Web Service**.
4. Connect your GitHub repository: `vickyvigneshcoc99-glitch/Student_performance_predictor`.
5. Use the following settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false`
   - **Instance Type**: Starter
6. Save and deploy.

Render will use the included `render.yaml` manifest automatically.

### Alternative: Docker deploy

If you prefer Docker, the repository already includes a `Dockerfile`.

---

## �📌 Notes for Internship Submission

- This is a self-contained mini-project — no external API keys or
  internet access required to run it.
- You can swap `student_dataset.csv` for real data with the same
  column names and everything will keep working.
- Model performance (R² ≈ 0.55 on synthetic data) is intentionally
  realistic — feel free to tune `train_model.py` (try `max_depth`,
  `n_estimators`, or a different model) to improve it.
