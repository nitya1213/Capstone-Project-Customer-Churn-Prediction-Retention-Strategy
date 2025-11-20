import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from preprocessing import prepare_for_modeling
from utils import save_model
import joblib

def tune_models():
    df_model, _ = prepare_for_modeling('data/customer_churn.csv', fit=True)
    X = df_model.drop(columns=['Churn (Target Variable)'])
    y = df_model['Churn (Target Variable)']

    tuned_models = {}

    # Logistic Regression tuning
    param_grid_lr = {'C': [0.01, 0.1, 1, 10], 'penalty':['l2'], 'solver':['lbfgs'], 'max_iter':[1000]}
    gs_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
    gs_lr.fit(X, y)
    tuned_models['LogisticRegression'] = gs_lr.best_estimator_
    print("LR best params:", gs_lr.best_params_, "best score:", gs_lr.best_score_)

    # Decision Tree tuning
    param_grid_dt = {'max_depth':[3,5,7,10,None], 'min_samples_leaf':[1,2,5,10]}
    gs_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='f1', n_jobs=-1)
    gs_dt.fit(X, y)
    tuned_models['DecisionTree'] = gs_dt.best_estimator_
    print("DT best params:", gs_dt.best_params_, "best score:", gs_dt.best_score_)

    # KNN tuning
    param_grid_knn = {'n_neighbors':[3,5,7,9], 'weights':['uniform','distance']}
    gs_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='f1', n_jobs=-1)
    gs_knn.fit(X, y)
    tuned_models['KNN'] = gs_knn.best_estimator_
    print("KNN best params:", gs_knn.best_params_, "best score:", gs_knn.best_score_)

    # Save best of all models by best cv score
    best_model = max([gs_lr, gs_dt, gs_knn], key=lambda gs: gs.best_score_).best_estimator_
    save_model(best_model, 'models/best_model.pkl')
    print("Saved best tuned model to models/best_model.pkl")
    return tuned_models

if __name__ == '__main__':
    tune_models()
