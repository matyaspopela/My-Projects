import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime

class DataProcessor:
    def __init__(self, data_source='all_perth_310121.csv'):
        self.data_source = data_source
    # Load data from CSV or DataFrame
    def load_data(self):
        if isinstance(self.data_source, str):
            self.data = pd.read_csv(self.data_source)
        elif isinstance(self.data_source, pd.DataFrame):
            self.data = self.data_source
        else:
            raise ValueError("Invalid data source")
        # One-Hot encode the string collumns
        categorical_cols = ['SUBURB', 'NEAREST_STN', 'NEAREST_SCH', ]
        self.data = pd.get_dummies(self.data, columns=categorical_cols)
        # sepearate key features/targets (PRICE, BEDROOMS, BUILD_YEAR, FLOOR_AREA, ) and target variable
        # clean the data
        self.clean_data()
        self.add_features()
        # select features and target after cleaning
        self.features = self.data.drop(columns=['PRICE', 'ADDRESS', 'DATE_SOLD', 'LATITUDE', 'LONGITUDE'])
        self.target = self.data['PRICE']

    def clean_data(self):
        # Drop rows with missing values in any of the relevant columns
        cols = ['BEDROOMS', 'BUILD_YEAR', 'FLOOR_AREA', 'LAND_AREA', 'BATHROOMS', 'GARAGE', 'NEAREST_STN_DIST', 'NEAREST_SCH_DIST']
        self.data = self.data.dropna(subset=cols).reset_index(drop=True)

    # normalize the data
    def normalize_data(self):
        #scaler from sklearn
        scaler = MinMaxScaler()
        #normalize
        self.features = pd.DataFrame(
            scaler.fit_transform(self.features),
            columns=self.features.columns
        )
        print("Data normalized")
    
    def split_data(self, train_size=0.7, va_size=0.15, test_size=0.15):
    # Ensure the split ratios sum to 1
        if train_size + va_size + test_size != 1.0:
            raise ValueError("Train, validation, and test sizes must sum to 1.0")
        # training and temp (validation + test) split
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.features, self.target, test_size=(va_size + test_size), random_state=42
        )
        # validation and test split
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(va_size)
        )
        # assign variables to self
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        print(f"Training set size: {len(self.X_train)}")
        print(f"Validation set size: {len(self.X_val)}")
        print(f"Testing set size: {len(self.X_test)}")
        
    def add_features(self):
        #dymanic current year
        current_year = datetime.datetime.now().year
        # add the age of the building as a feature
        self.data['AGE'] = current_year - self.data['BUILD_YEAR']
        self.data['IS_NEW'] = (self.data['AGE'] <= 20).astype(int)
        # now lets add garages per bedrooms, total_rooms
        self.data['GAREGEP_BEDROOMS'] = self.data['GARAGE'] / self.data['BEDROOMS']
        self.data['TOTAL_ROOMS'] = self.data['BEDROOMS'] + self.data['BATHROOMS']
        # add log transformed features
        self.data['LOG_LAND_AREA'] = np.log1p(self.data['LAND_AREA'])
        self.data['LOG_FLOOR_AREA'] = np.log1p(self.data['FLOOR_AREA'])
        self.data['LOG_CBD_DIST'] = np.log1p(self.data['CBD_DIST'])
        
        

        