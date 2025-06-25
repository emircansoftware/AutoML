import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, VotingClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from utils.before_model import prepare_features
def select_and_evaluate_models (problem_type: str = "",file_name: str = "",target_column: str = ""):
    """
    Problem türüne göre uygun modelleri seçer ve doğrulama skorlarını döndürür.
    Regresyon için R2, MAE, MSE; sınıflandırma için accuracy, F1 gibi metrikleri hesaplar.
    Ensemble modelleri de dahil edilmiştir.
    """
    X_train_transformed, X_test_transformed, y_train, y_test=prepare_features(file_name,target_column,problem_type)
    
    if problem_type == 'regression':
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'XGBRegressor': XGBRegressor(random_state=42),
            'SVR': SVR()
        }
        # Ensemble model
        ensemble = VotingRegressor(estimators=[
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor(random_state=42)),
            ('xgb', XGBRegressor(random_state=42)),
        ])
        models['Ensemble'] = ensemble
        results = {}
        for name, model in models.items():
            model.fit(X_train_transformed, y_train)
            y_pred = model.predict(X_test_transformed)
            results[name] = {
                'R2': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred)
            }
    elif problem_type == 'classification':
        models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'XGBClassifier': XGBClassifier(random_state=42),
            'SVC': SVC(random_state=42)
        }
        # Ensemble model
        ensemble = VotingClassifier(estimators=[
            ('lr', LogisticRegression(random_state=42)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('xgb', XGBClassifier(random_state=42))
        ])
        models['Ensemble'] = ensemble
        results = {}
        for name, model in models.items():
            model.fit(X_train_transformed, y_train)
            y_pred = model.predict(X_test_transformed)
            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred, average='weighted')
            }
    else:
        raise ValueError("Problem türü 'regression' veya 'classification' olmalıdır.")
    return results 