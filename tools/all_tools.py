from server import mcp
from utils.details import details
from utils.read_csv_file import read_csv
from utils.model_selection import select_and_evaluate_models
from utils.preprocessing import preprocessing
from utils.before_model import prepare_features
from utils.visualize_data import visualize_correlation_matrix_num, visualize_correlation_matrix_cat, visualize_correlation_matrix_final, visualize_outlier_detection, visualize_outlier_detection_final, visualize_confusion_matrix
from utils.hyperparameter import hyperparameter_tuning


@mcp.tool()
def information_about_data(file_name: str = ""):
    return details(file_name)


@mcp.tool()
def reading_csv(file_name: str = ""):
    return read_csv(file_name)

@mcp.tool()
def preprocessing_data(file_name: str = "", target_column: str = ""):
    return preprocessing(file_name, target_column)

@mcp.tool()
def prepare_data(file_name: str = "", target_column: str = "", problem_type: str = ""):
    return prepare_features(file_name, target_column, problem_type)

@mcp.tool()
def models(problem_type: str = "", file_name: str = "", target_column: str = ""):
    return select_and_evaluate_models(problem_type, file_name, target_column)

@mcp.tool()
def visualize_correlation_num(file_name: str = ""):
    return visualize_correlation_matrix_num(file_name)

@mcp.tool()
def visualize_correlation_cat(file_name: str = ""):
    return visualize_correlation_matrix_cat(file_name)

@mcp.tool()
def visualize_correlation_final(file_name: str = "",target_column: str = ""):
    return visualize_correlation_matrix_final(file_name,target_column)

@mcp.tool()
def visualize_outliers(file_name: str = ""):
    return visualize_outlier_detection(file_name)

@mcp.tool()
def visualize_outliers_final(file_name: str = "",target_column: str = ""):
    return visualize_outlier_detection_final(file_name,target_column)

@mcp.tool()
def visualize_accuracy_matrix(file_name: str = "",target_column: str = "",problem_type: str = ""):
    return visualize_confusion_matrix(file_name,target_column,problem_type)

@mcp.tool()
def best_model_hyperparameter(model_name, file_name, target_column, problem_type, n_trials, scoring, random_state):
    return hyperparameter_tuning(model_name, file_name, target_column, problem_type, n_trials, scoring, random_state)



