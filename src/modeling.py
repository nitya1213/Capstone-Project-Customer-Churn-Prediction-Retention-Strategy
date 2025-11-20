import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from preprocessing import prepare_for_modeling
from utils import save_model, load_csv
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

def build_and_evaluate():
    df_model, ids = prepare_for_modeling('data/customer_churn.csv', fit=True)

    # target column name in dataset: 'Churn (Target Variable)'
    y = df_model['Churn (Target Variable)']
    X = df_model.drop(columns=['Churn (Target Variable)'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=2000),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        cv = cross_val_score(model, X, y, cv=5, scoring='f1').mean()

        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'cv_f1': cv
        }

        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f} Precision: {prec:.4f} Recall: {rec:.4f} F1: {f1:.4f} CV-F1: {cv:.4f}")
        print(classification_report(y_test, preds))

        # save confusion matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.savefig(f'outputs/figures/{name}_confusion_matrix.png')
        plt.close()

    # pick best by CV-F1
    best_name = max(results.keys(), key=lambda k: results[k]['cv_f1'])
    best_model = results[best_name]['model']
    save_model(best_model, 'models/best_model.pkl')
    print("Best model:", best_name)
    return results

if __name__ == '__main__':
    build_and_evaluate()
