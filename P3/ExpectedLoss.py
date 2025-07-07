import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Load dataset
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# 2. Define features and target
X = df.drop(columns=['customer_id', 'default'])
y = df['default']

# 3. Split into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate model performance
y_proba = model.predict_proba(X_test_scaled)[:, 1]
auc_score = roc_auc_score(y_test, y_proba)
print(f"AUC Score: {auc_score:.4f}")

# 7. Prediction function for Expected Loss
def predict_expected_loss(credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding,
                          income, years_employed, fico_score):
    input_data = pd.DataFrame([[
        credit_lines_outstanding,
        loan_amt_outstanding,
        total_debt_outstanding,
        income,
        years_employed,
        fico_score
    ]], columns=X.columns)
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict probability of default
    pd_default = model.predict_proba(input_scaled)[0][1]
    
    # Calculate expected loss (Recovery Rate = 10%, so LGD = 90%)
    expected_loss = pd_default * loan_amt_outstanding * 0.9

    return {
        'Probability of Default (PD)': pd_default,
        'Expected Loss (EL)': expected_loss
    }

# 8. Example usage
example_result = predict_expected_loss(
    credit_lines_outstanding=2,
    loan_amt_outstanding=5000,
    total_debt_outstanding=7000,
    income=45000,
    years_employed=3,
    fico_score=600
)

print("\nExample Prediction:")
for k, v in example_result.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
