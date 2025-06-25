import pandas as pd
from utils.read_csv_file import read_csv
import numpy as np


def preprocessing(file_name: str = "",target_column: str = ""):
    print("target_column:", target_column, type(target_column))
    df=read_csv(file_name)
    null_summary = df.isnull().sum()
    print("Null değer sayısı:\n", null_summary)

    row_count = len(df)
    # Target column'daki null oranını kontrol et
    if df[target_column].isnull().sum() / row_count > 0.5:
        raise ValueError(f"Target column '{target_column}' null oranı %50'den fazla! Model eğitilemez.")

    high_null_cols = [col for col in df.columns if df[col].isnull().sum() / row_count > 0.5]
    if high_null_cols:
        print(f"%50'den fazla null içeren sütunlar siliniyor: {high_null_cols}")
        df = df.drop(columns=high_null_cols)

    # Kategorik sütunlarda 8'den fazla benzersiz değer varsa sütun sil
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 8]
    if high_cardinality_cols:
        print(f"8'den fazla benzersiz değer içeren kategorik sütunlar siliniyor: {high_cardinality_cols}")
        df = df.drop(columns=high_cardinality_cols)

    # Kategorik sütunlarda, değer oranı %5'ten az olan satırları sil
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col == target_column:
            continue
        value_counts = df[col].value_counts(normalize=True)
        low_ratio_values = value_counts[value_counts < 0.1].index
        if not low_ratio_values.empty:
            print(f"{col} sütununda %5'ten az orana sahip değerler siliniyor: {low_ratio_values}")
            df = df[~df[col].isin(low_ratio_values)]

    # ID gibi sıralı değerleri sil
    id_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    if id_cols:
        print(f"ID gibi sıralı değerler siliniyor: {id_cols}")
        df = df.drop(columns=id_cols)

    # Çok fazla outlier olan satırları sil
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_column:
            continue
        col_data = df[col].dropna()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Oranlı sınırlar
        extreme_lower = lower * 0.4
        extreme_upper = upper * 1.6

        outlier_mask = (df[col] < extreme_lower) | (df[col] > extreme_upper)
        if outlier_mask.any():
            print(f"{col} sütununda çok uç outlier olan satırlar siliniyor.")
            df = df[~outlier_mask]

    df = df.dropna(subset=[target_column])

    # Null doldurma işlemleri
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Kategorik: En çok tekrar edeni doldur
                most_frequent = df[col].mode()[0]
                df[col] = df[col].fillna(most_frequent)
            else:
                # Numerik: Outlier kontrolü ve uygun doldurma
                col_data = df[col].dropna()
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_ratio = ((col_data < lower) | (col_data > upper)).mean()
                if outlier_ratio < 0.3:
                    # Outlier azsa ortalama ile doldur
                    mean_val = col_data.mean()
                    df[col] = df[col].fillna(mean_val)
                else:
                    # Outlier fazlaysa medyan ile doldur
                    median_val = col_data.median()
                    df[col] = df[col].fillna(median_val)
    return df 