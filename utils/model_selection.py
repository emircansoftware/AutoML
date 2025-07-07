import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, VotingClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from utils.before_model import prepare_features
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostRegressor, CatBoostClassifier

def select_and_evaluate_models(problem_type: str = "", file_name: str = "", target_column: str = ""):
    """
    Problem türüne göre uygun modelleri seçer ve doğrulama skorlarını döndürür.
    Regresyon için R2, MAE, MSE; sınıflandırma için accuracy, F1 gibi metrikleri hesaplar.
    Ensemble modelleri de dahil edilmiştir.
    Ayrıca en iyi modelin adını, nesnesini ve tahmin sonuçlarını da döndürür.
    """
    X_train_transformed, X_test_transformed, y_train, y_test = prepare_features(file_name, target_column, problem_type)

    best_model_name = None
    best_model = None
    best_score = -np.inf
    best_y_pred = None

    if problem_type == 'regression':
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'XGBRegressor': XGBRegressor(random_state=42),
            'SVR': SVR(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'CatBoostRegressor': CatBoostRegressor(verbose=0, random_state=42)
        }
        # Ensemble model
        ensemble = VotingRegressor(estimators=[
            ('lr', LinearRegression()),
            ('ridge', Ridge()),
            ('rf', RandomForestRegressor(random_state=42)),
            ('xgb', XGBRegressor(random_state=42)),
            ('cat', CatBoostRegressor(verbose=0, random_state=42))
        ])
        models['Ensemble'] = ensemble
        results = {}
        for name, model in models.items():
            model.fit(X_train_transformed, y_train)
            y_pred = model.predict(X_test_transformed)
            r2 = r2_score(y_test, y_pred)
            results[name] = {
                'R2': r2,
                'MAE': mean_absolute_error(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred)
            }
            if r2 > best_score:
                best_score = r2
                best_model_name = name
                best_model = model
                best_y_pred = y_pred
    elif problem_type == 'classification':
        models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'RidgeClassifier': RidgeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'XGBClassifier': XGBClassifier(random_state=42),
            'SVC': SVC(random_state=42),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
            'GaussianNB': GaussianNB(),
            'CatBoostClassifier': CatBoostClassifier(verbose=0, random_state=42)
        }
        # Ensemble model
        ensemble = VotingClassifier(estimators=[
            ('lr', LogisticRegression(random_state=42)),
            ('ridge', RidgeClassifier()),
            ('rf', RandomForestClassifier(random_state=42)),
            ('xgb', XGBClassifier(random_state=42)),
            ('cat', CatBoostClassifier(verbose=0, random_state=42))
        ])
        models['Ensemble'] = ensemble
        results = {}
        for name, model in models.items():
            model.fit(X_train_transformed, y_train)
            y_pred = model.predict(X_test_transformed)
            acc = accuracy_score(y_test, y_pred)
            results[name] = {
                'Accuracy': acc,
                'F1': f1_score(y_test, y_pred, average='weighted')
            }
            if acc > best_score:
                best_score = acc
                best_model_name = name
                best_model = model
                best_y_pred = y_pred
    else:
        raise ValueError("Problem türü 'regression' veya 'classification' olmalıdır.")
    
    # En iyi modelin sonuçlarını içeren dictionary döndür
    final_results = {
        'results': results,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'y_test': y_test,
        'y_pred': best_y_pred
    }
    
    return final_results 