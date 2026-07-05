<div align="center">

# Trackly

### Smart Student Performance Prediction

**AI-powered academic risk prediction system for early student intervention**

*AI Application Programming — CAI 360*
*Princess Nourah bint Abdulrahman University, College of Computer Science and Information*

</div>

---

## Overview

**Trackly** is an AI-powered decision-support system that helps academic advisors identify students who may be at risk of poor academic performance — **before** the semester ends and formal warnings are issued.

Instead of waiting for final grades to reveal struggling students, Trackly uses early-semester indicators (attendance, missed assignments, midterm scores, and coursework performance) to flag at-risk students early, giving advisors time to actually intervene.

The system is built as a complete end-to-end AI application: from data preparation and model training, to a FastAPI backend and a web dashboard that advisors can use directly — no coding required.

> ⚠️ Trackly is a **decision-support tool**, not an automated decision-maker. The final advising decision always remains with the human advisor.

---

## Key Features

- **Single-student prediction** — enter a student's record manually and get an instant risk assessment
- **Batch prediction via Excel upload** — screen hundreds of students at once from a spreadsheet
- **Risk probability score** — not just a label, but a confidence score to help prioritize outreach
- **Course demand forecasting** — a second model that predicts future enrollment per major/semester, useful for department-level planning
- **Explainable predictions** — feature importance is reported so advisors understand why a student was flagged
- **Privacy-first design** — trained on synthetic data, with no student identifiers used in modeling
- **LLM-powered recommendations** *(implemented, pending activation)* — generates supportive next-step suggestions for advisors using Groq's LLaMA 3.3 70B model

---

## System Architecture

Trackly follows a simple client–server architecture:

```
Advisor → Dashboard (HTML) → FastAPI Backend → ML Pipeline (joblib) → Prediction + Probability
```

The trained model (`academic_risk_model.joblib`) bundles **preprocessing and classification together** in a single scikit-learn Pipeline, so the backend never has to manually encode or scale input — it just builds a DataFrame and calls `.predict()`.

---

## Machine Learning Components

### 1. Academic Risk Classifier

A **Random Forest classifier** predicts whether a student is *At Risk* or *Not at Risk* using early academic indicators only (no final grades — those are excluded to prevent data leakage).

| Configuration | Value |
|---|---|
| Model | Random Forest (`n_estimators=200, max_depth=8, min_samples_split=10, min_samples_leaf=4`) |
| Training data | 20,000 synthetic student-semester records |
| Class balancing | SMOTE-NC (applied to training set only) |
| Tuning | GridSearchCV, 5-fold CV, F1-scoring |

**Test-set performance:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.801 | 0.614 | 0.817 | 0.701 | 0.894 |
| **Random Forest (deployed)** | **0.814** | **0.637** | **0.810** | **0.713** | **0.897** |
| Tuned Random Forest | 0.813 | 0.636 | 0.810 | 0.713 | 0.896 |

**Top predictive features:** attendance rate (36.4%), midterm score (32.9%), coursework score (20.3%), missed assignments (7.9%).

### 2. Course Demand Forecasting

A **Random Forest Regressor** estimates expected course enrollment per (semester, major) group, supporting department-level planning.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 13.73 | 18.43 | 0.78 |
| **Random Forest Regressor** | **6.88** | **7.82** | **0.96** |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Modeling | scikit-learn, imbalanced-learn (SMOTE-NC), Pandas, NumPy |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | HTML / CSS / JavaScript (single-page dashboard) |
| Model serialization | joblib |
| LLM recommendations | Groq API (LLaMA 3.3 70B) |
| Deployment | Render (Python 3.12) |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Basic health check |
| `GET` | `/app` | Serves the web dashboard |
| `GET` | `/health` | Returns model status and valid input options |
| `POST` | `/predict` | Single-student risk prediction |
| `POST` | `/upload` | Excel batch prediction |

---

## Dataset

Trackly is trained on a **synthetic** student performance dataset (generated due to privacy restrictions on real institutional data), structured across three linked tables:

- **Students** — ~5,000 students × 4 semesters = 20,000 records
- **Courses** — course-level data across 7 CS-related majors
- **Enrollment** — ~60,000 student–course enrollment records

Input features used for prediction: `attendance_rate_pct`, `missed_assignments`, `midterms_total_30`, `coursework_30`, `semester`, `major`.

> Features that would leak final outcomes (`GPA`, `cumulative_GPA`, `final_exam_40`, `term_grade_100`, `student_id`) were deliberately excluded to keep the model usable as a genuine *early-warning* tool.

---

## Ethical Considerations

- **Privacy:** synthetic data only; no student identifiers used in modeling
- **Fairness:** at-risk prediction rates checked across majors (26.5%–30.3%) — no extreme bias detected
- **Human oversight:** Trackly surfaces students for review; it never issues penalties or final academic decisions
- **Transparency:** feature importance is reported so predictions can be explained to advisors

---

## Known Limitations

- Trained on **synthetic** data with a known rule-based labeling scheme — real-world validation is still needed
- At the default threshold, the model misses ~19% of truly at-risk students (recall = 0.810)
- Regression component trained on only 28 (semester, major) groups — treat as proof-of-concept
- LLM recommendation module is implemented but not yet activated in the live deployment (pending API key configuration)
- Render free-tier hosting introduces cold-start delays (~40s) after inactivity

---

## Future Improvements

- Integrate real, anonymized institutional data
- Explore gradient-boosted models (XGBoost, LightGBM) and decision-threshold tuning
- Activate the Groq LLM recommendation feature
- Move to a paid hosting tier to eliminate cold starts
- Add temporal features (attendance/score trends over time)

---

## Team

| Name |
|---|
| Layan Alfares |
| Shmokh Aljomah |
| Dana Alsubaie |
| Deem Alrashoud |
| Maha Alhudaybi |

*AI Application Programming (CAI 360) — Princess Nourah bint Abdulrahman University*

---

## References

Key libraries and platforms used: scikit-learn, imbalanced-learn, Pandas, FastAPI, Uvicorn, Groq API, and Render. See the full technical report for complete citations.
