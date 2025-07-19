# Auto ML - Automated Machine Learning Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-1.9.4+-orange.svg)](https://modelcontextprotocol.io/)

An intelligent automated machine learning platform that provides comprehensive data analysis, preprocessing, model selection, and hyperparameter tuning capabilities through Model Context Protocol (MCP) tools.

## 🚀 Features

### 📊 Data Analysis & Exploration

- **Data Information**: Get comprehensive dataset statistics including shape, memory usage, data types, and missing values
- **CSV Reading**: Efficient CSV file reading with pandas and pyarrow support
- **Correlation Analysis**: Visualize correlation matrices for numerical and categorical variables
- **Outlier Detection**: Identify and visualize outliers in your datasets

### 🔧 Data Preprocessing

- **Automated Preprocessing**: Handle missing values, encode categorical variables, and scale numerical features
- **Feature Engineering**: Prepare features for both regression and classification problems
- **Data Validation**: Check for duplicates and data quality issues

### 🤖 Machine Learning Models

- **Multiple Algorithms**: Support for various ML algorithms including:
  - **Regression**: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, XGBoost, SVR, KNN, CatBoost
  - **Classification**: Logistic Regression, Ridge Classifier, Random Forest, XGBoost, SVM, KNN, Decision Tree, Naive Bayes, CatBoost

### 📈 Model Evaluation & Visualization

- **Performance Metrics**:
  - Regression: R², MAE, MSE
  - Classification: Accuracy, F1-Score
- **Confusion Matrix Visualization**: For classification problems
- **Model Comparison**: Compare multiple models side-by-side

### ⚙️ Hyperparameter Tuning

- **Automated Tuning**: Optimize model hyperparameters using advanced search algorithms
- **Customizable Scoring**: Choose from various evaluation metrics
- **Trial Management**: Control the number of optimization trials

## 📁 Project Structure

```
AutoML/
├── data/                   # Sample datasets
│   ├── Ai.csv
│   ├── Calories.csv
│   ├── Cost.csv
│   ├── Digital.csv
│   ├── Electricity.csv
│   ├── ford.csv
│   ├── Habits.csv
│   ├── heart.csv
│   ├── Lifestyle.csv
│   ├── Mobiles.csv
│   ├── Personality.csv
│   ├── Salaries.csv
│   ├── Shopper.csv
│   ├── Sleep.csv
│   ├── cat.csv
│   ├── test.csv
│   └── train.csv
├── tools/
│   └── all_tools.py       # MCP tool definitions
├── utils/
│   ├── before_model.py        # Feature preparation
│   ├── details.py             # Data information
│   ├── external_test.py       # External data test with XGBoost
│   ├── feature_importance.py  # Feature importance analysis
│   ├── hyperparameter.py      # Hyperparameter tuning
│   ├── model_selection.py     # Model selection and evaluation
│   ├── prediction.py          # Prediction utilities
│   ├── preprocessing.py       # Data preprocessing
│   ├── read_csv_file.py       # CSV reading utilities
│   └── visualize_data.py      # Visualization functions
├── main.py                # Application entry point
├── server.py              # MCP server configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip or uv package manager

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/emircansoftware/AutoML.git
   cd AutoML
   ```

2. **Install dependencies**

   ```bash
   # Using pip
   pip install -r requirements.txt
   pip install uv

   ```

## Using with Claude Desktop

### 1. Data Path Setting

In `utils/read_csv_file.py`, update the `path` variable to match your own project directory on your computer:

```python
# Example:
path = r"C:\\YOUR\\PROJECT\\PATH\\AutoML\\data"
```

### 2. Claude Desktop Configuration

In Claude Desktop, add the following block to your `claude_desktop_config.json` file and adjust the paths to match your own system:

```json
{
  "mcpServers": {
    "AutoML": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\YOUR\\PROJECT\\PATH\\AutoML",
        "run",
        "main.py"
      ]
    }
  }
}
```

You can now start your project from Claude Desktop.

## 📋 Dependencies

- **MCP Framework**: `mcp[cli]>=1.9.4` - Model Context Protocol for tool integration
- **Data Processing**: `pandas>=2.3.0`, `pyarrow>=20.0.0`, `numpy>=2.3.1`
- **Machine Learning**: `scikit-learn>=1.3.0`, `xgboost>=2.0.0`, `lightgbm>=4.3.0`
- **Additional ML**: `catboost` (for CatBoost models)

## 🎯 Usage

### Starting the MCP Server

```python
from server import mcp

# Run the server
mcp.run()
```

### Available Tools

The platform provides the following MCP tools:

#### Data Analysis Tools

- `information_about_data(file_name)`: Give detailed information about the data
- `reading_csv(file_name)`: Read the csv file
- `visualize_correlation_num(file_name)`: Visualize the correlation matrix for numerical columns
- `visualize_correlation_cat(file_name)`: Visualize the correlation matrix for categorical columns
- `visualize_correlation_final(file_name, target_column)`: Visualize the correlation matrix after preprocessing
- `visualize_outliers(file_name)`: Visualize outliers in the data
- `visualize_outliers_final(file_name, target_column)`: Visualize outliers after preprocessing

#### Preprocessing Tools

- `preprocessing_data(file_name, target_column)`: Preprocess the data (remove outliers, fill nulls, etc.)
- `prepare_data(file_name, target_column, problem_type)`: Prepare the data for models (encoding, scaling, etc.)

#### Model Training & Evaluation

- `models(problem_type, file_name, target_column)`: Select and evaluate models based on problem type
- `visualize_accuracy_matrix(file_name, target_column, problem_type)`: Visualize the confusion matrix for predictions
- `best_model_hyperparameter(model_name, file_name, target_column, problem_type, n_trials, scoring, random_state)`: Tune the hyperparameters of the best model
- `test_external_data(main_file_name, target_column, problem_type, test_file_name)`: Test external data with the best model and return predictions
- `predict_value(model_name, file_name, target_column, problem_type, n_trials, scoring, random_state, input)`: Predict the value of the target column for new input
- `feature_importance_analysis(file_name, target_column, problem_type)`: Analyze the feature importance of the data using XGBoost

### Example Workflow

```python
# 1. Analyze your data
info = information_about_data("data/heart.csv")

# 2. Preprocess the data
preprocessed = preprocessing_data("data/heart.csv", "target")

# 3. Prepare features for classification
features = prepare_data("data/heart.csv", "target", "classification")

# 4. Train and evaluate models
results = models("classification", "data/heart.csv", "target")

# 5. Visualize results
confusion_matrix = visualize_accuracy_matrix("data/heart.csv", "target", "classification")

# 6. Optimize best model
best_model = best_model_hyperparameter("RandomForestClassifier", "data/heart.csv", "target", "classification", 100, "accuracy", 42)
```

## 📊 Sample Datasets (All CSV datasets are from Kaggle.)

The project includes various sample datasets for testing:

- **heart.csv**: Heart disease prediction dataset
- **Salaries.csv**: Salary prediction dataset
- **Calories.csv**: Calorie prediction dataset
- **Personality.csv**: Personality analysis dataset
- **Digital.csv**: Digital behavior dataset
- **Lifestyle.csv**: Lifestyle analysis dataset
- **Mobiles.csv**: Mobile phone dataset
- **Habits.csv**: Habit analysis dataset
- **Sleep.csv**: Sleep pattern dataset
- **Cost.csv**: Cost analysis dataset
- **ford.csv**: Ford car dataset
- **Ai.csv**: AI-related dataset
- **cat.csv**: Cat-related dataset

## 🔧 Configuration

### Environment Variables

- Set your preferred random seed for reproducible results
- Configure MCP server settings in `server.py`

### Customization

- Add new ML algorithms in `utils/model_selection.py`
- Extend preprocessing steps in `utils/preprocessing.py`
- Create custom visualization functions in `utils/visualize_data.py`

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/) for the MCP framework
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting
- [CatBoost](https://catboost.ai/) for categorical boosting
- [pandas](https://pandas.pydata.org/) for data manipulation

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/emircansoftware/MCP_Server_DataScience/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

---

