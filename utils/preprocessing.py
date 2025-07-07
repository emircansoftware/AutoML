import pandas as pd
from utils.read_csv_file import read_csv
import numpy as np


def preprocessing(file_name: str = "",target_column: str = ""):
    df=read_csv(file_name)
    null_summary = df.isnull().sum()
    
    # İşlem bilgilerini saklamak için dictionary
    process_info = {
        'initial_rows': len(df),
        'null_summary': null_summary.to_dict(),
        'warnings': [],
        'removed_columns': [],
        'removed_rows': 0
    }

    row_count = len(df)
    # Numeric sütunlar arasında 8'den az unique değere sahip olanların tipini 'category' olarak değiştir
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    low_unique_cols = [col for col in numeric_cols if df[col].nunique() < 8]
    for col in low_unique_cols:
        df[col] = df[col].astype('category')

    # Target column'daki null değerleri kontrol et ve güvenli şekilde temizle
    target_null_count = df[target_column].isnull().sum()
    if target_null_count > 0:
        if target_null_count / len(df) > 0.1:  # %10'dan fazla null varsa uyarı ver
            process_info['warnings'].append(f"Target column'da %{(target_null_count/len(df)*100):.1f} oranında null değer var!")
        
        # Null değerleri olan satırları sil
        initial_len = len(df)
        df = df.dropna(subset=[target_column])
        process_info['removed_rows'] += initial_len - len(df)
        
        # Eğer çok az veri kaldıysa uyarı ver
        if len(df) < 10:
            process_info['warnings'].append("Çok az veri kaldı! Model eğitimi için yeterli veri olmayabilir.")

    high_null_cols = [col for col in df.columns if df[col].isnull().sum() / row_count > 0.5]
    if high_null_cols:
        process_info['removed_columns'].extend(high_null_cols)
        df = df.drop(columns=high_null_cols)

    # Kategorik sütunlarda 8'den fazla benzersiz değer varsa sütun sil
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 8]
    if high_cardinality_cols:
        process_info['removed_columns'].extend(high_cardinality_cols)
        df = df.drop(columns=high_cardinality_cols)

    # Kategorik sütunlarda, değer oranı %5'ten az olan satırları sil
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col == target_column:
            continue
        value_counts = df[col].value_counts(normalize=True)
        low_ratio_values = value_counts[value_counts < 0.1].index
        if not low_ratio_values.empty:
            initial_len = len(df)
            df = df[~df[col].isin(low_ratio_values)]
            process_info['removed_rows'] += initial_len - len(df)

    # ID gibi sıralı değerleri sil
    id_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    if id_cols:
        process_info['removed_columns'].extend(id_cols)
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
            initial_len = len(df)
            df = df[~outlier_mask]
            process_info['removed_rows'] += initial_len - len(df)

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
    
    process_info['final_rows'] = len(df)
    process_info['final_columns'] = list(df.columns)
    
    return df, process_info 