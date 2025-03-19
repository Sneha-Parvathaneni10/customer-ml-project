import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('/content/customer_churn_data.csv')
X = data.drop('Churn', axis=1)
y = data['Churn']

X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print("Classification Report:")
print(classification_report(y_test, y_pred))
original_features = ['tenure', 'Dependents', 'MonthlyCharges', 'TotalCharges']
new_data_df = pd.DataFrame(new_data, columns=original_features)
new_data_encoded = pd.get_dummies(new_data_df, columns=['Dependents']) # Assuming only 'Dependents' was one-hot encoded
new_data_encoded = new_data_encoded.reindex(columns=X.columns, fill_value=0)
new_data_scaled = scaler.transform(new_data_encoded)
prediction = model.predict(new_data_scaled)
