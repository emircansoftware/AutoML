import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.read_csv_file import read_csv
from utils.preprocessing import preprocessing
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from utils.model_selection import select_and_evaluate_models
from sklearn.preprocessing import LabelEncoder

def visualize_correlation_matrix_num(file_name: str = ""):
    df=read_csv(file_name)
    numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix=df[numeric_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',center=0)
    plt.title('Numeric Column Correlation Matrix')
    plt.show()

def visualize_correlation_matrix_cat(file_name: str = ""):
    df = read_csv(file_name)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[cat_col], y=df[num_col])
            plt.title(f'{num_col} dağılımı - {cat_col} kategorilerine göre')
            plt.xlabel(cat_col)
            plt.ylabel(num_col)
            plt.show()

def visualize_correlation_matrix_final(file_name: str = "",target_column: str = ""):
    df, process_info = preprocessing(file_name,target_column)
    numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols=df.select_dtypes(include=['object', 'category']).columns.tolist()
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in categorical_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    corr_matrix = df_encoded[numeric_cols+categorical_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',center=0)
    plt.title('Final Correlation Matrix (Encoded)')
    plt.show()

def visualize_outlier_detection(file_name: str = ""):
    df=read_csv(file_name)
    numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        plt.figure(figsize=(10,8))
        sns.boxplot(x=df[col])
        plt.title(f'{col} Outlier Detection')
        plt.show()

def visualize_outlier_detection_final(file_name: str = "",target_column: str = ""):
    df, process_info = preprocessing(file_name,target_column)
    numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        plt.figure(figsize=(10,8))
        sns.boxplot(x=df[col])
        plt.title(f'{col} Outlier Detection')
        plt.show()
        
   
def visualize_confusion_matrix(file_name: str = "",target_column: str = "",problem_type: str = ""):
    final_results = select_and_evaluate_models(problem_type,file_name,target_column)
    y_test = final_results['y_test']
    y_pred = final_results['y_pred']
    cm = sk_confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm,annot=True,cmap='coolwarm',center=0)
    plt.title('Confusion Matrix')
    plt.show()
    

    
