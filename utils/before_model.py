# Ortak yardımcı fonksiyonlar burada tanımlanacak 

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from utils.preprocessing import preprocessing

def prepare_features(file_name: str = "",target_column: str = "",problem_type: str = ""):
    """
    Numerik ve kategorik sütunları ayırt eder, uygun scaler ve encoder'ları uygular.
    Numerik sütunlarda outlier oranına göre RobustScaler veya StandardScaler kullanır.
    Kategorik sütunlarda, değer sayısı 3'ten az ise OneHotEncoder, değilse OrdinalEncoder kullanır.
    Problem türü sınıflandırma ise target sütununu LabelEncoder ile dönüştürür.
    Problem türü regresyon ise y değerlerini log1p ile dönüştürür.
    Train-test split yapar ve ColumnTransformer'ı uygular.
    """
    df=preprocessing(file_name,target_column)

    print("Sütunlar:", df.columns.tolist())
    print("target_column:", target_column, type(target_column))

    # Numerik ve kategorik sütunları ayırt et
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Numerik sütunlar için scaler seçimi
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

    # Kategorik sütunlar için encoder seçimi
    categorical_transformer = []
    for col in categorical_cols:
        if df[col].nunique() <= 3:  # 3 veya daha az benzersiz değer varsa
            categorical_transformer.append((f'onehot_{col}', OneHotEncoder(sparse_output=False), [col]))
        else:  # 3'ten fazla benzersiz değer varsa
            categorical_transformer.append((f'ordinal_{col}', OrdinalEncoder(), [col]))

    # ColumnTransformer oluştur
    preprocessor = ColumnTransformer(
        transformers=numeric_transformer + categorical_transformer,
        remainder='passthrough'
    )

    # Target sütununu dönüştür
    if problem_type == 'classification':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
    elif problem_type == 'regression':
        df[target_column] = np.log1p(df[target_column])


    print("İşlemlerden sonraki df:",df.head())

    # Train-test split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ColumnTransformer'ı uygula
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, X_test_transformed, y_train, y_test 