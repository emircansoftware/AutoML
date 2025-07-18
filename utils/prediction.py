from utils.hyperparameter import hyperparameter_tuning
import numpy as np
import pandas as pd
from utils.read_csv_file import read_csv
from utils.before_model import prepare_features_with_preprocessor
from utils.preprocessing import preprocessing

def get_user_input(file_name, target_column):
    # Preprocessing sonrası kalan feature sütunlarını al  # Get remaining feature columns after preprocessing
    df_processed, process_info = preprocessing(file_name, target_column)
    feature_columns = [col for col in df_processed.columns if col != target_column]
    print("Lütfen aşağıdaki sırayla, değerleri virgül ile ayırarak girin:  # Please enter the values in the following order, separated by commas:")
    print(", ".join(feature_columns))
    user_input = input("Değerleri girin:  # Enter the values:")  # Örnek: 63, 1, 145, 233, erkek, A sınıfı  # Example: 63, 1, 145, 233, male, Class A
    user_values = []
    for val in user_input.split(","):
        val = val.strip()  # baştaki ve sondaki boşlukları sil  # Remove leading and trailing spaces
        try:
            if '.' in val:
                val = float(val)
            else:
                val = int(val)
        except ValueError:
            pass  # string olarak bırak  # Leave as string
        user_values.append(val)
    return np.array([user_values], dtype=object), feature_columns

def predict_value(model_name, file_name, target_column, problem_type, n_trials=3, scoring=None, random_state=42, input=None, use_log1p=False):
    # Eğitimdeki preprocessor ve labelencoder'ı al  # Get preprocessor and labelencoder from training
    if problem_type == 'classification':
        _, _, _, _, preprocessor, le = prepare_features_with_preprocessor(file_name, target_column, problem_type)
    else:
        _, _, _, _, preprocessor = prepare_features_with_preprocessor(file_name, target_column, problem_type)
    
    best_model, params, best_score = hyperparameter_tuning(model_name, file_name, target_column, problem_type, n_trials, scoring, random_state)
    
    # Preprocessing sonrası kalan feature sütunlarını al  # Get remaining feature columns after preprocessing
    df_processed, process_info = preprocessing(file_name, target_column)
    feature_columns = [col for col in df_processed.columns if col != target_column]

    # Kullanıcıdan input al ve DataFrame'e çevir  # Get input from user and convert to DataFrame
    if input is None:
        input, feature_columns = get_user_input(file_name, target_column)
        input_df = pd.DataFrame(input, columns=feature_columns)
    elif isinstance(input, str):
        user_values = [v.strip() for v in input.split(",")]
        input_df = pd.DataFrame([user_values], columns=feature_columns)
        # Numerik kolonları uygun şekilde dönüştür  # Convert numeric columns appropriately
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except ValueError:
                pass
    else:
        # Eğer input zaten DataFrame veya array ise, feature isimleriyle DataFrame'e çevir  # If input is already a DataFrame or array, convert to DataFrame with feature names
        input_df = pd.DataFrame(input, columns=feature_columns)
    
    # Preprocessing: eğitimdeki preprocessor ile encode et  # Preprocessing: encode with preprocessor from training
    input_transformed = preprocessor.transform(input_df)
    
    # Tahmin  # Prediction
    prediction = best_model.predict(input_transformed)
    if problem_type == 'classification':
        prediction = le.inverse_transform(prediction)
    elif problem_type == 'regression':
        prediction = np.expm1(prediction)
    return prediction







