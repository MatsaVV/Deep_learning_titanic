import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
from azureml.core import Workspace
from azureml.core.model import Model

# Se connecter à l'espace de travail Azure ML
ws = Workspace.from_config("src/config.json")

# Charger les données
df = pd.read_csv('data/Titanic.csv')

# Préparation des données
features = df[['sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'alone']]
target = df['survived']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

numeric_features = ['age', 'sibsp', 'parch', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['sex', 'embarked', 'class', 'who', 'alone']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [10, 20]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Suivre l'expérience avec MLflow
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment('Titanic_Survival_Prediction_Comparison')

with mlflow.start_run():
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", grid_search.best_score_)

# Enregistrer le modèle dans Azure ML
model_path = "models/best_model.pkl"
mlflow.sklearn.save_model(best_model, model_path)
Model.register(workspace=ws, model_name="titanic_model", model_path=model_path)
