import pandas as pd
from utils.read_csv_file import read_csv
import numpy as np


def preprocessing(file_name: str = "",target_column: str = ""):
    df=read_csv(file_name)
    null_summary = df.isnull().sum()
    
    # Dictionary to store process information
    process_info = {
        'initial_rows': len(df),
        'null_summary': null_summary.to_dict(),
        'warnings': [],
        'removed_columns': [],
        'removed_rows': {
            'target_null': 0,
            'low_ratio': 0,
            'outliers': 0,
            'total': 0
        },
        'debug_info': {}  # For debug information
    }

    row_count = len(df)
    print(f"Initial row count: {row_count}") 

    df=df.drop_duplicates()

    
    # Change type to 'category' for numeric columns with less than 11 unique values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    low_unique_cols = [col for col in numeric_cols if df[col].nunique() < 11]
    for col in low_unique_cols:
        df[col] = df[col].astype('category')

    # Check and safely remove null values in the target column
    target_null_count = df[target_column].isnull().sum()
    if target_null_count > 0:
        if target_null_count / len(df) > 0.1:  # Warn if more than 10% null
            process_info['warnings'].append(f"There are {(target_null_count/len(df)*100):.1f}% null values in the target column!")
        
        # Drop rows with null values in the target column
        initial_len = len(df)
        df = df.dropna(subset=[target_column])
        removed = initial_len - len(df)
        process_info['removed_rows']['target_null'] = removed
        process_info['removed_rows']['total'] += removed
        process_info['debug_info']['target_null_removed'] = removed
        print(f"Target null values removed: {removed} rows")

    high_null_cols = [col for col in df.columns if df[col].isnull().sum() / row_count > 0.5]
    if high_null_cols:
        process_info['removed_columns'].append({'column': high_null_cols, 'reason': 'high_null_ratio'})
        df = df.drop(columns=high_null_cols)
        print(f"Columns with high null ratio removed: {high_null_cols}")

    # Drop categorical columns with more than 10 unique values
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 10]
    if high_cardinality_cols:
        process_info['removed_columns'].append({'column': high_cardinality_cols, 'reason': 'high_cardinality'})
        df = df.drop(columns=high_cardinality_cols)
        print(f"High cardinality columns removed: {high_cardinality_cols}")

    # Remove rows with values less than 10% in categorical columns
    total_removed_low_ratio = 0
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col == target_column:
            continue
        value_counts = df[col].value_counts(normalize=True)
        low_ratio_values = value_counts[value_counts < 0.1].index
        if not low_ratio_values.empty:
            initial_len = len(df)
            df = df[~df[col].isin(low_ratio_values)]
            removed = initial_len - len(df)
            total_removed_low_ratio += removed
            process_info['removed_rows']['low_ratio'] += removed
            print(f"In column {col}, low ratio values removed: {removed} rows, values: {list(low_ratio_values)}")
    
    process_info['debug_info']['low_ratio_removed'] = total_removed_low_ratio

    # Remove sequential values like ID (only those containing 'id' in the name and not the target_column)
    id_cols = [
        col for col in df.columns
        if 'id' in col.lower() and df[col].nunique() == len(df) and col != target_column
    ]
    if id_cols:
        process_info['removed_columns'].append({'column': id_cols, 'reason': 'id_like'})
        df = df.drop(columns=id_cols)
        print(f"ID-like columns removed: {id_cols}")

    # Remove rows with too many outliers
    total_removed_outliers = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_column:
            continue
        col_data = df[col].dropna()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Proportional limits - correct calculation for negative values
        if lower < 0:
            # For negative values: make it even more negative (outlier)
            extreme_lower = lower * 1.6
        else:
            # For positive values: make it even more positive (outlier)
            extreme_lower = lower * 0.4
            
        if upper < 0:
            # For negative values: make it less negative (outlier)
            extreme_upper = upper * 0.4
        else:
            # For positive values: make it even more positive (outlier)
            extreme_upper = upper * 1.6

        outlier_mask = (df[col] < extreme_lower) | (df[col] > extreme_upper)
        if outlier_mask.any():
            initial_len = len(df)
            df = df[~outlier_mask]
            removed = initial_len - len(df)
            total_removed_outliers += removed
            process_info['removed_rows']['outliers'] += removed
            print(f"Outliers removed in column {col}: {removed} rows")
    
    process_info['debug_info']['outliers_removed'] = total_removed_outliers

    # Fill missing values
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Categorical: Fill with the most frequent value
                most_frequent = df[col].mode()[0]
                df[col] = df[col].fillna(most_frequent)
            else:
                # Numeric: Outlier check and appropriate filling
                col_data = df[col].dropna()
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_ratio = ((col_data < lower) | (col_data > upper)).mean()
                if outlier_ratio < 0.3:
                    # If few outliers, fill with mean
                    mean_val = col_data.mean()
                    df[col] = df[col].fillna(mean_val)
                else:
                    # If many outliers, fill with median
                    median_val = col_data.median()
                    df[col] = df[col].fillna(median_val)
    
    process_info['final_rows'] = len(df)
    process_info['final_columns'] = list(df.columns)
    
    print(f"Final row count: {len(df)}")
    print(f"Total removed rows: {process_info['removed_rows']['total']}")
    print(f"Debug info: {process_info['debug_info']}")
    
    return df, process_info 