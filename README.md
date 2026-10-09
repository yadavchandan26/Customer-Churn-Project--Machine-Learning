# Telco Customer Churn Prediction

A simple machine learning project that predicts whether a telecom customer will leave (churn) or stay, using Logistic Regression.

## Dataset

- **Name:** Telco Customer Churn (IBM sample dataset, available on Kaggle)
- **Rows:** about 7,043 customers
- **Target column:** `Churn` (Yes / No)
- **Class balance:** roughly 26% churn, so the data is imbalanced

## Project Structure

```
churn_project/
├── main.py                      # Entry point, runs the full pipeline
├── data/
│   └── telco_churn.csv          # Dataset
├── prediction/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Load, clean, encode, split
│   ├── model_training.py        # Build and train the model
│   └── model_evaluation.py      # Metrics and submission file
├── notebook.ipynb               # Exploration notebook
└── README.md
```

## How It Works

1. **Load data** from `data/telco_churn.csv`.
2. **Preprocess**
   - Drop `customerID` (text ID, not useful for the model).
   - Convert `TotalCharges` to numbers (blank values become 0).
   - Convert Yes/No columns and `gender` to 1/0.
   - One-hot encode `Contract`, `InternetService`, `PaymentMethod`.
3. **Split** into 70% train and 30% test (`random_state=42`, stratified on `Churn`).
4. **Train** a Logistic Regression model with `class_weight='balanced'`, with `StandardScaler` in a Pipeline.
5. **Evaluate** on test data: accuracy, ROC-AUC, precision, recall, F1.
6. **Save predictions** to `submission.csv`.

## Setup

Python 3.9 or newer is recommended.

```bash
pip install pandas scikit-learn
```

## Run

From inside the `churn_project/` folder:

```bash
python main.py
```

Always run from the project root, otherwise `from prediction...` imports will fail.

## Output

- Metrics printed in the terminal (train and test accuracy, ROC-AUC, classification report).
- `submission.csv` with the predicted `Churn` value (1 = will churn, 0 = will stay) for each test row.

## Expected Results

Logistic Regression on this dataset usually gives about **75-80% test accuracy** and about **75-80% recall** on the churn class. If you see 90% or more, check for data leakage.


## Model Comparison

Three models were compared on the same 70/30 split (test set = 2,113 customers). Decision Tree and Random Forest were tuned with GridSearchCV.

| Model | Accuracy | Churn Recall | Churn Precision | Churn F1 |
|---|---|---|---|---|
| Logistic Regression (balanced) | 0.76 | 0.84 | 0.53 | **0.65** |
| Decision Tree (balanced, tuned) | 0.73 | 0.83 | 0.50 | 0.63 |
| Random Forest (tuned) | **0.80** | 0.45 | 0.70 | 0.54 |

Logistic Regression was chosen because it gives the best churn F1 and the highest recall, so it catches the most customers who are about to leave. Random Forest has higher accuracy, but it misses more than half of the churners. Logistic Regression is also simpler and easier to explain.

Full experiments are in `notebook.ipynb`.

## Author

Yadav Chandan R.
