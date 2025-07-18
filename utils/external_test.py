from utils.before_model import prepare_features_with_preprocessor
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from utils.read_csv_file import read_csv
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_log_error , accuracy_score
from utils.hyperparameter import hyperparameter_tuning
from utils.preprocessing import preprocessing


def test_external_data_xgb(main_file_name: str = "",target_column: str = "",problem_type: str = "",test_file_name: str = ""):

    # Preprocess the main data and get the preprocessor
    if problem_type == "classification":
        X_train_transformed, X_test_transformed, y_train, y_test, preprocessor, label_encoder = prepare_features_with_preprocessor(main_file_name, target_column, problem_type)
    else:
        X_train_transformed, X_test_transformed, y_train, y_test, preprocessor = prepare_features_with_preprocessor(main_file_name, target_column, problem_type)
    
    # Combine numpy arrays
    X = np.vstack([X_train_transformed, X_test_transformed])
    y = pd.concat([y_train, y_test]).reset_index(drop=True)
    
    # Read the test data
    test_df = read_csv(test_file_name)
    
    # Use the same columns as the main data (except target column)
    main_df, _ = preprocessing(main_file_name, target_column)
    main_features = main_df.drop(columns=[target_column]).columns
    
    # Check for missing columns in test data and add them
    missing_cols = set(main_features) - set(test_df.columns)
    for col in missing_cols:
        test_df[col] = 0  # Fill missing columns with 0
    
    # Arrange test data columns in the same order as main data
    test_processed = test_df[main_features]
    
    # Apply the same preprocessing to the test data
    test = preprocessor.transform(test_processed)

    FOLDS = 3
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    # Empty array for out-of-fold predictions
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test))
    test_preds_proba = np.zeros((len(test), len(y.unique()))) if problem_type == "classification" else None
    
    # Hyperparameter optimization (done once)
    print("Hyperparameter optimization is starting...")
    if problem_type == "classification":
        best_model, best_params, study_best_value = hyperparameter_tuning("XGBClassifier", main_file_name, target_column, "classification", n_trials=1, scoring=None, random_state=42)
    else:
        best_model, best_params, study_best_value = hyperparameter_tuning("XGBRegressor", main_file_name, target_column, "regression", n_trials=1, scoring=None, random_state=42)
    print(f"Best parameters: {best_params}")
    print(f"Best score: {study_best_value:.4f}")



    if problem_type=="classification":
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            print(f"\nFold {fold}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBClassifier(
                **best_params,
                early_stopping_rounds=25
            )

            start = time.time()

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100
            )


            val_preds = model.predict(X_val)
            oof_preds[val_idx] = val_preds


            test_preds_proba += model.predict_proba(test)

            acc = accuracy_score(y_val, val_preds)
            print(f" Fold {fold} Accuracy: {acc:.4f}")
            print(f" Time: {time.time() - start:.1f} sec")


        test_preds_proba /= FOLDS
        test_preds = np.argmax(test_preds_proba, axis=1)

        # Convert numeric predictions to original class labels
        test_preds_original = label_encoder.inverse_transform(test_preds)

        oof_acc = accuracy_score(y, oof_preds)
        print(f"\n Final OOF Accuracy: {oof_acc:.4f}")
        print(f"\n Final Test Predictions (numeric): {test_preds}")
        print(f"\n Final Test Predictions (original labels): {test_preds_original}")
        return test_preds_original
    else:
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            print(f"\nFold {fold}")

            # Train / validation data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Apply log1p transformation
            y_train_log = np.log1p(y_train)
            y_val_log = np.log1p(y_val)

            model = XGBRegressor(
                **best_params,
                early_stopping_rounds=25
            )

            start = time.time()

            model.fit(
                X_train, y_train_log,
                eval_set=[(X_val, y_val_log)],
                verbose=100
            )

            # Predictions
            val_preds_log = model.predict(X_val)
            test_preds += model.predict(test)

            # Save OOF log predictions
            oof_preds[val_idx] = val_preds_log

            # Fold score (log RMSE)
            rmse = np.sqrt(((val_preds_log - y_val_log) ** 2).mean())
            print(f"Fold {fold} Log-RMSE: {rmse:.4f}")
            print(f"Time: {time.time() - start:.1f} seconds")

        # Average test prediction
        test_preds /= FOLDS
        test_preds = np.expm1(test_preds)

        # Convert back from log to real values
        final_oof = np.expm1(oof_preds)

        # RMSLE (real error)
        rmsle = np.sqrt(mean_squared_log_error(y, final_oof))
        print(f"\nFinal RMSLE: {rmsle:.4f}")
        print(f"\nFinal Test Predictions: {test_preds}")
        return test_preds