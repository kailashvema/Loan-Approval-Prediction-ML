import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Load dataset
df = pd.read_csv("loan_data.csv")

# Encode categorical columns
encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])


# Features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Logistic Regression
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, pred1))


# Random Forest
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, pred2))


# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred2))


# Feature Importance
importance = model2.feature_importances_
features = X.columns

plt.barh(features, importance)
plt.title("Feature Importance")
plt.show()