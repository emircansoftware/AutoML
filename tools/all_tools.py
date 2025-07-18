from server import mcp
from utils.details import details
from utils.read_csv_file import read_csv
from utils.model_selection import select_and_evaluate_models
from utils.preprocessing import preprocessing
from utils.before_model import prepare_features
from utils.visualize_data import visualize_correlation_matrix_num, visualize_correlation_matrix_cat, visualize_correlation_matrix_final, visualize_outlier_detection, visualize_outlier_detection_final, visualize_confusion_matrix
from utils.hyperparameter import hyperparameter_tuning
from utils.external_test import test_external_data_xgb
from utils.prediction import predict_value
from utils.feature_importance import feature_importance


@mcp.tool(
    name="information_about_data",
    description="Give detailed information about the data"
)
def information_about_data(file_name: str = ""):
    return details(file_name)


@mcp.tool(
    name="read_csv_file",
    description="Read the csv file"
)
def reading_csv(file_name: str = ""):
    return read_csv(file_name)

@mcp.tool(
    name="preprocessing_data",
    description="Preprocess the data such as remove outliers, not affected columns,fill null values, etc."
)
def preprocessing_data(file_name: str = "", target_column: str = ""):
    return preprocessing(file_name, target_column)

@mcp.tool(
    name="prepare_data",
    description="Prepare the data for models such as encoding, scaling, etc."
)
def prepare_data(file_name: str = "", target_column: str = "", problem_type: str = ""):
    return prepare_features(file_name, target_column, problem_type)

@mcp.tool(
    name="select_and_evaluate_models",
    description="Depend on the problem type, select the best model and evaluate the models"
)
def models(problem_type: str = "", file_name: str = "", target_column: str = ""):
    return select_and_evaluate_models(problem_type, file_name, target_column)

@mcp.tool(
    name="visualize_correlation_numbers",
    description="If user wants to see the correlation between the numerical columns, this tool will visualize the correlation matrix"
)
def visualize_correlation_num(file_name: str = ""):
    return visualize_correlation_matrix_num(file_name)

@mcp.tool(
    name="visualize_correlation_categories",
    description="If user wants to see the correlation between the categorical columns, this tool will visualize the correlation matrix"
)
def visualize_correlation_cat(file_name: str = ""):
    return visualize_correlation_matrix_cat(file_name)

@mcp.tool(
    name="visualize_correlation_final",
    description="If user wants to see the correlation between the columns after preprocessing, this tool will visualize the correlation matrix"
)
def visualize_correlation_final(file_name: str = "",target_column: str = ""):
    return visualize_correlation_matrix_final(file_name,target_column)

@mcp.tool(
    name="visualize_outliers",
    description="If user wants to see the outliers in the data, this tool will visualize the outliers"
)
def visualize_outliers(file_name: str = ""):
    return visualize_outlier_detection(file_name)

@mcp.tool(
    name="visualize_outliers_final",
    description="If user wants to see the outliers in the data after preprocessing, this tool will visualize the outliers"
)
def visualize_outliers_final(file_name: str = "",target_column: str = ""):
    return visualize_outlier_detection_final(file_name,target_column)

@mcp.tool(
    name="visualize_accuracy_matrix",
    description="If user wants to see the accuracy of the model, this tool will visualize the prediction and actual values in a confusion matrix"
)
def visualize_accuracy_matrix(file_name: str = "",target_column: str = "",problem_type: str = ""):
    return visualize_confusion_matrix(file_name,target_column,problem_type)

@mcp.tool(
    name="best_model_hyperparameter",
    description="Tune the hyperparameters of the best model"
)
def best_model_hyperparameter(model_name, file_name, target_column, problem_type, n_trials, scoring, random_state):
    return hyperparameter_tuning(model_name, file_name, target_column, problem_type, n_trials, scoring, random_state)

@mcp.tool(
    name="test_external_data",
    description="Test the external data with the best model and return the prediction"
)
def test_external_data(main_file_name: str = "",target_column: str = "",problem_type: str = "",test_file_name: str = ""):
    return test_external_data_xgb(main_file_name,target_column,problem_type,test_file_name)

@mcp.tool(
    name="predict_value",
    description="Predict the value of the target column"
)
def predict_value_tool(model_name, file_name, target_column, problem_type, n_trials, scoring, random_state, input):
    return predict_value(model_name, file_name, target_column, problem_type, n_trials, scoring, random_state, input)

@mcp.tool(
    name="feature_importance_analysis",
    description="Analyze the feature importance of the data using XGBoost"
)
def feature_importance_analysis(file_name: str = "", target_column: str = "", problem_type: str = ""):
    return feature_importance(file_name, target_column, problem_type)





