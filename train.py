import numpy as np
from data import DataProcessor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from data import DataProcessor
import warnings
from sklearn.exceptions import ConvergenceWarning
import joblib

# Suppress ConvergenceWarning and UserWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# load processor
processor = DataProcessor()

def fit(X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(n_estimators=50,max_depth=None, max_features=None, random_state=42)
    model.fit(X_train, y_train)
    # evaluate
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    #calculate median error
    median_price = processor.data['PRICE'].median()
    average_price = processor.data['PRICE'].sum() / len(processor.data['PRICE'])
    diff = average_price - median_price
    error = np.sqrt(val_mse)
    error_percent_median = ((error-diff) / median_price) * 100
    error_percent_average = (error / average_price) * 100
    print(f"Error (average price): ${error:.2f} ({error_percent_average:.2f}%)")
    print(f"Error (median price): ${error - diff:.2f} ({error_percent_median:.2f}%)")
    return model


def main():
    #load data via class
    
    processor.load_data()
    processor.normalize_data()
    processor.split_data()
    # convert our data into np.arrays
    X_train = processor.X_train
    y_train = processor.y_train
    X_val = processor.X_val.values
    y_val = processor.y_val.values
    
    #create & fit model for features !
    model = fit(X_train, y_train, X_val, y_val)
    # save the model
    joblib.dump(model, 'current_model.pk1')
    

# python dela tohle (if __name__ == "__main__": main())
# kdyz mas vic main funkci v projektu ale chces sputit tuhle konkretni.
if __name__ == "__main__":
    main()
    