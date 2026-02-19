import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# ---------------------------------------------------
# 1. Create Dataset Manually (No CSV Required)
# ---------------------------------------------------
data = {
    'Age': [25, 45, 35, 22, 40, 28, 50, 30, 27, 48],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Pages_Viewed': [5, 10, 7, 3, 12, 6, 15, 4, 8, 11],
    'Time_Spent': [10, 25, 15, 5, 30, 12, 35, 8, 18, 22],
    'Purchase': [0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)
# ---------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------
# Handle missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)
# Encode categorical variable (Gender)
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])  # Male=1, Female=0 (or vice versa)
# Separate features and target
X = df.drop('Purchase', axis=1)
y = df['Purchase']
# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ---------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
# ---------------------------------------------------
# 4. Train Naive Bayes Model
# ---------------------------------------------------
model = GaussianNB()
model.fit(X_train, y_train)
# ---------------------------------------------------
# 5. Model Evaluation
# ---------------------------------------------------
y_pred = model.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
# ---------------------------------------------------
# 6. Predict for New Customer
# ---------------------------------------------------
# Example: Age=32, Gender=Female, Pages=8, Time=18
new_customer = np.array([[32, 0, 8, 18]])  # Gender encoded
new_customer_scaled = scaler.transform(new_customer)
probability = model.predict_proba(new_customer_scaled)
print("\nProbability of Not Purchasing:", probability[0][0])
print("Probability of Purchasing    :", probability[0][1])