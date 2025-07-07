import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

# 1. Load dataset
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# 2. Prepare features and target
X = df.drop(columns=['customer_id', 'default'])
y = df['default']

# 3. Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train classifiers
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# 6. Evaluate performance
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
lr_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled)[:, 1])
print(f"Random Forest AUC: {rf_auc:.4f}")
print(f"Logistic Regression AUC: {lr_auc:.4f}")

# 7. Use the better model for predictions
final_model = rf_model if rf_auc >= lr_auc else lr_model
print("Using model:", "Random Forest" if final_model == rf_model else "Logistic Regression")

# 8. Function to estimate Probability of Default and Expected Loss
def estimate_expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score,
    model=final_model
):
    input_df = pd.DataFrame([[
        credit_lines_outstanding,
        loan_amt_outstanding,
        total_debt_outstanding,
        income,
        years_employed,
        fico_score
    ]], columns=X.columns)

    input_scaled = scaler.transform(input_df)
    pd_value = model.predict_proba(input_scaled)[0][1]
    expected_loss = pd_value * loan_amt_outstanding * 0.9  # LGD = 90%

    return {
        "Probability of Default (PD)": pd_value,
        "Expected Loss (EL)": expected_loss
    }

# 9. Example usage
example = estimate_expected_loss(
    credit_lines_outstanding=3,
    loan_amt_outstanding=8000,
    total_debt_outstanding=12000,
    income=60000,
    years_employed=5,
    fico_score=580
)
print("\nExample Borrower Prediction:")
for key, val in example.items():
    print(f"{key}: {val:.4f}")

# 10. Predict and export test set with actual and predicted values
X_test_copy = X_test.copy()
X_test_copy['actual_default'] = y_test.values
X_test_copy['predicted_default'] = final_model.predict(X_test_scaled)

X_test_copy.to_csv("predicted_defaults_test_set.csv", index=False)
print("\nTest set predictions saved to 'predicted_defaults_test_set.csv'")
