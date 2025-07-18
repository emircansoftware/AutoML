import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from utils.before_model import prepare_features
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostRegressor, CatBoostClassifier

def select_and_evaluate_models(problem_type: str = "", file_name: str = "", target_column: str = ""):
    
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
        results = {}
        for name, model in models.items():
            try:
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
            except Exception as e:
                print(f"Model {name} error: {e}")
                continue
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
        results = {}
        for name, model in models.items():
            try:
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
            except Exception as e:
                print(f"Model {name} error: {e}")
                continue
    else:
        raise ValueError("Problem type must be 'regression' or 'classification'.")
    
    # Return a dictionary containing the results of the best model
    final_results = {
        'results': results,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'y_test': y_test,
        'y_pred': best_y_pred
    }
    
    return final_results

def train_single_model(model_name: str, X_train, y_train, problem_type: str = "regression"):
    
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
    else:
        raise ValueError("Problem type must be 'regression' or 'classification'.")
    
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not supported. Supported models: {list(models.keys())}")
    
    model = models[model_name]
    model.fit(X_train, y_train)
    return model 