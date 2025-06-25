import pandas as pd
from utils.read_csv_file import read_csv


def details(file_name: str = ""):
    df = read_csv(file_name)
    
    # Temel bilgiler
    columns = df.columns
    first_row = df.head(1)
    null_values = df.isnull().sum()
    
    # Ek bilgiler
    shape = df.shape
    data_types = df.dtypes
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB cinsinden
    
    # Sayısal sütunlar için istatistikler
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_stats = ""
    if len(numeric_columns) > 0:
        numeric_stats = f"\nSayısal sütunların istatistikleri:\n{df[numeric_columns].describe()}"
    
    # Kategorik sütunlar için bilgiler
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    categorical_info = ""
    if len(categorical_columns) > 0:
        categorical_info = "\nKategorik sütunların benzersiz değer sayıları:\n"
        for col in categorical_columns:
            unique_count = df[col].nunique()
            categorical_info += f"{col}: {unique_count} benzersiz değer\n"
    
    # Duplicate satırlar
    duplicate_count = df.duplicated().sum()
    
    result = f"""
Veri Seti Bilgileri:
===================
Dosya: {file_name}
Boyut: {shape[0]} satır, {shape[1]} sütun
Bellek kullanımı: {memory_usage:.2f} MB
Duplicate satır sayısı: {duplicate_count}

Sütunlar: {list(columns)}
Veri tipleri: {dict(data_types)}

İlk satır:
{first_row}

Eksik değerler:
{null_values}
{categorical_info}{numeric_stats}
"""
    
    return result


