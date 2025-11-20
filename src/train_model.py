import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from data_preprocessing import preprocess_data
from utils import load_data

df = load_data('data/customer_churn.csv')

df_processed = preprocess_data(df)
X = df_processed.drop('Churn (Target Variable)', axis=1)
y = df_processed['Churn (Target Variable)']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    print(f"{name} Accuracy:", accuracy_score(Y_test, pred))
    print(classification_report(Y_test, pred))

best_model = models['Logistic Regression']
joblib.dump(best_model, 'models/churn_model.pkl')

print("Best model saved!")
