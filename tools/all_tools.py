from server import mcp
from utils.details import details
from utils.read_csv_file import read_csv
from utils.model_selection import select_and_evaluate_models
from utils.preprocessing import preprocessing
from utils.before_model import prepare_features


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



