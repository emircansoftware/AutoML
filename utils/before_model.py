# Ortak yardımcı fonksiyonlar burada tanımlanacak  # Common helper functions will be defined here

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from utils.preprocessing import preprocessing

def prepare_features(file_name: str = "",target_column: str = "",problem_type: str = ""):
    
    df, process_info = preprocessing(file_name,target_column)

    # Distinguish between numerical and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Select scaler for numerical columns
    numeric_transformer = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_ratio = ((col_data < lower) | (col_data > upper)).mean()
        if outlier_ratio > 0.3:
            numeric_transformer.append((f'robust_{col}', RobustScaler(), [col]))
        else:
            numeric_transformer.append((f'standard_{col}', StandardScaler(), [col]))

    # Select encoder for categorical columns
    categorical_transformer = []
    for col in categorical_cols:
        if df[col].nunique() <= 3:  # If 3 or fewer unique values
            categorical_transformer.append((f'onehot_{col}', OneHotEncoder(sparse_output=False), [col]))
        else:  # If more than 3 unique values
            categorical_transformer.append((f'ordinal_{col}', OrdinalEncoder(), [col]))

    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=numeric_transformer + categorical_transformer,
        remainder='passthrough'
    )

    # Transform target column
    if problem_type == 'classification':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
    elif problem_type == 'regression':
        df[target_column] = np.log1p(df[target_column])

    # Train-test split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply ColumnTransformer
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, X_test_transformed, y_train, y_test

def prepare_features_with_preprocessor(file_name: str = "",target_column: str = "",problem_type: str = ""):
    df, process_info = preprocessing(file_name,target_column)

    # Distinguish between numerical and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Select scaler for numerical columns
    numeric_transformer = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_ratio = ((col_data < lower) | (col_data > upper)).mean()
        if outlier_ratio > 0.3:
            numeric_transformer.append((f'robust_{col}', RobustScaler(), [col]))
        else:
            numeric_transformer.append((f'standard_{col}', StandardScaler(), [col]))

    # Select encoder for categorical columns
    categorical_transformer = []
    for col in categorical_cols:
        if df[col].nunique() <= 3:  # If 3 or fewer unique values
            categorical_transformer.append((f'onehot_{col}', OneHotEncoder(sparse_output=False), [col]))
        else:  # If more than 3 unique values
            categorical_transformer.append((f'ordinal_{col}', OrdinalEncoder(), [col]))

    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=numeric_transformer + categorical_transformer,
        remainder='passthrough'
    )

    # Transform target column
    if problem_type == 'classification':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
    elif problem_type == 'regression':
        df[target_column] = np.log1p(df[target_column])

    # Train-test split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply ColumnTransformer
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if problem_type == 'classification':
        return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor, le
    else:
        return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor 

def prepare_features_with_names(file_name: str = "", target_column: str = "", problem_type: str = ""):
    df, process_info = preprocessing(file_name, target_column)

    # Distinguish between numerical and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Select scaler for numerical columns
    numeric_transformer = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_ratio = ((col_data < lower) | (col_data > upper)).mean()
        if outlier_ratio > 0.3:
            numeric_transformer.append((f'robust_{col}', RobustScaler(), [col]))
        else:
            numeric_transformer.append((f'standard_{col}', StandardScaler(), [col]))

    # Select encoder for categorical columns
    categorical_transformer = []
    for col in categorical_cols:
        if df[col].nunique() <= 3:
            categorical_transformer.append((f'onehot_{col}', OneHotEncoder(sparse_output=False), [col]))
        else:
            categorical_transformer.append((f'ordinal_{col}', OrdinalEncoder(), [col]))

    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=numeric_transformer + categorical_transformer,
        remainder='passthrough'
    )

    # Transform target column
    if problem_type == 'classification':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
    elif problem_type == 'regression':
        df[target_column] = np.log1p(df[target_column])

    # Train-test split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply ColumnTransformer
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get correct feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # For old scikit-learn versions
        feature_names = [str(i) for i in range(X_train_transformed.shape[1])]

    return X_train_transformed, X_test_transformed, y_train, y_test, feature_names 