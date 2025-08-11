import numpy as np
from data import DataProcessor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from data import DataProcessor
import warnings
from sklearn.exceptions import ConvergenceWarning


# Suppress ConvergenceWarning and UserWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def hyper_parameters():
    processor = DataProcessor()
    processor.load_data()
    processor.normalize_data()
    processor.split_data()
    X_train = processor.X_train
    y_train = processor.y_train
    X_val = processor.X_val.values
    y_val = processor.y_val.values

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': [None, 'sqrt', 'log2']
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    y_val_pred = grid.best_estimator_.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    print("Best params:", grid.best_params_)
    print(f"Results of best estimator: ${np.sqrt(val_mse):.2f}")

hyper_parameters()

#note: 157k best param(utilizes max performance.)