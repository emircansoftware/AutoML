import pandas as pd
from utils.read_csv_file import read_csv


def details(file_name: str = ""):
    df = read_csv(file_name)
    
    # Basic information
    columns = df.columns
    first_row = df.head(1)
    null_values = df.isnull().sum()
    
    # Additional information
    shape = df.shape
    data_types = df.dtypes
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # In MB
    
    # Statistics for numerical columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_stats = ""
    if len(numeric_columns) > 0:
        numeric_stats = f"\nStatistics for numerical columns:\n{df[numeric_columns].describe()}"
    
    # Information for categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    categorical_info = ""
    if len(categorical_columns) > 0:
        categorical_info = "\nUnique value counts for categorical columns:\n"
        for col in categorical_columns:
            unique_count = df[col].nunique()
            categorical_info += f"{col}: {unique_count} unique values\n"
    
    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    
    result = f"""
Data information:
===================
File: {file_name}
Size: {shape[0]} rows, {shape[1]} columns
Memory usage: {memory_usage:.2f} MB
Duplicate rows: {duplicate_count}

Columns: {list(columns)}
Data types: {dict(data_types)}

First row:
{first_row}

Missing values:
{null_values}
{categorical_info}{numeric_stats}
"""
    
    return result


