# 🏥 Diabetes Prediction – AI Mini Project

## Setup Instructions

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap streamlit joblib
```

### 2. Download the dataset
- Go to: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Download `diabetes.csv` and place it in this folder

### 3. Run the full pipeline
```bash
python diabetes_prediction_pipeline.py
```
This will generate:
- `eda_distributions.png`
- `eda_heatmap.png`
- `eda_boxplots.png`
- `eval_confusion_matrices.png`
- `eval_roc_curves.png`
- `insights_feature_importance.png`
- `insights_shap.png`
- `model_xgb.pkl`, `scaler.pkl`, `feature_names.pkl`

### 4. Launch the Streamlit app
```bash
streamlit run streamlit_app.py
```

## Project Structure
```
├── diabetes_prediction_pipeline.py  # Full ML pipeline (Sections 2–8)
├── streamlit_app.py                 # Deployment app (Section 9)
├── diabetes.csv                     # Dataset (download from Kaggle)
├── README.md                        # This file
└── *.png                            # Generated plots
```

## Models Used
| Model | Type |
|---|---|
| Logistic Regression | Baseline classifier |
| Random Forest | Ensemble (with GridSearch) |
| XGBoost | Gradient boosting (best model) |

## Team
- Group members here
