import numpy as np
from utils.before_model import prepare_features
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostRegressor, CatBoostClassifier

# Model type-specific parameter grids
param_grids = {
    'RandomForestRegressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    },
    'XGBRegressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.5, 1.0],
        'colsample_bytree': [0.5, 1.0],
        'gamma': [0, 5],
        'reg_alpha': [0, 1],
        'reg_lambda': [0, 1]
    },
    'XGBClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.5, 1.0],
        'colsample_bytree': [0.5, 1.0],
        'gamma': [0, 5],
        'reg_alpha': [0, 1],
        'reg_lambda': [0, 1]
    },
    'SVR': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 5]
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 5]
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs', 'saga'],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'l1_ratio': [0, 1]
    },
    'LinearRegression': {},
    'Ridge': {
        'alpha': [0.01, 10.0]
    },
    'Lasso': {
        'alpha': [0.01, 10.0]
    },
    'ElasticNet': {
        'alpha': [0.01, 10.0],
        'l1_ratio': [0, 1]
    },
    'KNeighborsRegressor': {
        'n_neighbors': [2, 20],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    'RidgeClassifier': {
        'alpha': [0.01, 10.0]
    },
    'KNeighborsClassifier': {
        'n_neighbors': [2, 20],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    'DecisionTreeClassifier': {
        'max_depth': [2, 20],
        'min_samples_split': [2, 10],
        'criterion': ['gini', 'entropy', 'log_loss']
    },
    'GaussianNB': {},
    'CatBoostRegressor': {
        'iterations': [100, 1000],
        'depth': [3, 10],
        'learning_rate': [0.01, 0.3],
        'l2_leaf_reg': [1, 10],
        'random_state': [None],
        'verbose': [0]
    },
    'CatBoostClassifier': {
        'iterations': [100, 1000],
        'depth': [3, 10],
        'learning_rate': [0.01, 0.3],
        'l2_leaf_reg': [1, 10],
        'random_state': [None],
        'verbose': [0]
    }
}

def hyperparameter_tuning(model_name, file_name, target_column, problem_type, n_trials=3, scoring=None, random_state=42):
    
    n_trials = int(n_trials)
    random_state = int(random_state)
    X_train, X_test, y_train, y_test = prepare_features(file_name, target_column, problem_type)
    if scoring is None:
        scoring = 'r2' if problem_type == 'regression' else 'accuracy'

    def objective(trial):
        if model_name == 'RandomForestRegressor':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': random_state
            }
            mdl = RandomForestRegressor(**params)
        elif model_name == 'RandomForestClassifier':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': random_state
            }
            mdl = RandomForestClassifier(**params)
        elif model_name == 'XGBRegressor':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': random_state
            }
            mdl = XGBRegressor(**params)
        elif model_name == 'XGBClassifier':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': random_state
            }
            mdl = XGBClassifier(**params)
        elif model_name == 'SVR':
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            params = {
                'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                'kernel': kernel,
                'gamma': trial.suggest_float('gamma', 0.001, 1.0),
            }
            if kernel == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
            mdl = SVR(**params)
        elif model_name == 'SVC':
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            params = {
                'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                'kernel': kernel,
                'gamma': trial.suggest_float('gamma', 0.001, 1.0),
                'random_state': random_state
            }
            if kernel == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
            mdl = SVC(**params)
        elif model_name == 'LogisticRegression':
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
            solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga'])
            params = {
                'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                'penalty': penalty,
                'solver': solver,
                'random_state': random_state
            }
            if penalty == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
            mdl = LogisticRegression(**params)
        elif model_name == 'LinearRegression':
            mdl = LinearRegression()
            params = {}
        elif model_name == 'Ridge':
            params = {
                'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True)
            }
            mdl = Ridge(**params)
        elif model_name == 'Lasso':
            params = {
                'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True)
            }
            mdl = Lasso(**params)
        elif model_name == 'ElasticNet':
            params = {
                'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0, 1)
            }
            mdl = ElasticNet(**params)
        elif model_name == 'KNeighborsRegressor':
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 2, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2)
            }
            mdl = KNeighborsRegressor(**params)
        elif model_name == 'RidgeClassifier':
            params = {
                'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True)
            }
            mdl = RidgeClassifier(**params)
        elif model_name == 'KNeighborsClassifier':
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 2, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2)
            }
            mdl = KNeighborsClassifier(**params)
        elif model_name == 'DecisionTreeClassifier':
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            }
            mdl = DecisionTreeClassifier(**params)
        elif model_name == 'GaussianNB':
            params = {}
            mdl = GaussianNB()
        elif model_name == 'CatBoostRegressor':
            params = {
                'iterations': trial.suggest_int('iterations', 300, 2000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_state': random_state,
                'verbose': 0
            }
            mdl = CatBoostRegressor(**params)
        elif model_name == 'CatBoostClassifier':
            params = {
                'iterations': trial.suggest_int('iterations', 300, 2000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_state': random_state,
                'verbose': 0
            }
            mdl = CatBoostClassifier(**params)
        else:
            raise ValueError(f"{model_name} is not defined in Optuna search space.")

        if problem_type == 'regression':
            score = cross_val_score(mdl, X_train, y_train, scoring='r2', cv=3).mean()
        else:
            score = cross_val_score(mdl, X_train, y_train, scoring='accuracy', cv=3).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    # Train the model with the best parameters
    if model_name == 'RandomForestRegressor':
        best_model = RandomForestRegressor(**best_params, random_state=random_state)
    elif model_name == 'RandomForestClassifier':
        best_model = RandomForestClassifier(**best_params, random_state=random_state)
    elif model_name == 'XGBRegressor':
        best_model = XGBRegressor(**best_params, random_state=random_state)
    elif model_name == 'XGBClassifier':
        best_model = XGBClassifier(**best_params, random_state=random_state)
    elif model_name == 'SVR':
        best_model = SVR(**best_params)
    elif model_name == 'SVC':
        best_model = SVC(**best_params, random_state=random_state)
    elif model_name == 'LogisticRegression':
        best_model = LogisticRegression(**best_params, random_state=random_state)
    elif model_name == 'LinearRegression':
        best_model = LinearRegression()
    elif model_name == 'Ridge':
        best_model = Ridge(**best_params)
    elif model_name == 'Lasso':
        best_model = Lasso(**best_params)
    elif model_name == 'ElasticNet':
        best_model = ElasticNet(**best_params)
    elif model_name == 'KNeighborsRegressor':
        best_model = KNeighborsRegressor(**best_params)
    elif model_name == 'RidgeClassifier':
        best_model = RidgeClassifier(**best_params)
    elif model_name == 'KNeighborsClassifier':
        best_model = KNeighborsClassifier(**best_params)
    elif model_name == 'DecisionTreeClassifier':
        best_model = DecisionTreeClassifier(**best_params)
    elif model_name == 'GaussianNB':
        best_model = GaussianNB()
    elif model_name == 'CatBoostRegressor':
        best_model = CatBoostRegressor(**best_params, random_state=random_state, verbose=0)
    elif model_name == 'CatBoostClassifier':
        best_model = CatBoostClassifier(**best_params, random_state=random_state, verbose=0)
    else:
        raise ValueError(f"{model_name} model could not be created.")

    best_model.fit(X_train, y_train)
    return best_model, best_params, study.best_value
