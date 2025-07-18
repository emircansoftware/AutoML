from utils.before_model import prepare_features_with_names
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor


def feature_importance(file_name: str = "", target_column: str = "", problem_type: str = ""):
    
    X_train_transformed, X_test_transformed, y_train, y_test, feature_names = prepare_features_with_names(
        file_name=file_name,
        target_column=target_column,
        problem_type=problem_type
    )
    X = np.vstack([X_train_transformed, X_test_transformed])
    X_df = pd.DataFrame(X, columns=feature_names)
    y = pd.concat([y_train, y_test]).reset_index(drop=True)

    if problem_type == "classification":
        model = XGBClassifier()
    else:
        model = XGBRegressor()

    model.fit(X_df, y)
    importance = model.feature_importances_
    features = X_df.columns

    # Simplify feature names
    simple_features = [f.split('__')[-1] for f in features]

    # Visualization
    plt.figure(figsize=(10, max(5, len(simple_features) * 0.4)))  # Set height based on feature count
    plt.barh(simple_features, importance)
    plt.xlabel("Importance Score")
    plt.title("Feature Importance")
    plt.yticks(fontsize=10)  # Reduce font size
    plt.tight_layout()       # Automatic layout
    plt.show()
    
