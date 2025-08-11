import joblib as jlb
import numpy as np
import pandas as pd
from data import DataProcessor

class Predictor:
    def __init__(self, data_file='all_perth_310121.csv', model_file='current_model.pk1'):
        self.data_file = data_file
        self.model_file = model_file
        self.processor = DataProcessor(self.data_file)
        self.model = None
        self.df = None
        self.target = None
        self.original_data = None

    def load_and_prepare_data(self):
        """Load and prepare data for predictions"""
        if self.df is None:  # Only load once
            # Load original data for display purposes
            self.original_data = pd.read_csv(self.data_file)
            
            # Process data for predictions
            self.processor.load_data()
            self.processor.clean_data()
            self.processor.add_features()
            self.processor.normalize_data()
            self.df = self.processor.features
            self.target = self.processor.target

    def get_houses_list(self):
        """Get list of houses with basic info for web app dropdown"""
        try:
            self.load_and_prepare_data()
            print(f"Data loaded. Features shape: {self.df.shape}, Target shape: {self.target.shape}")
            
            houses = []
            # Limit to first 100 houses for faster loading
            max_houses = min(100, len(self.df))
            
            for i in range(max_houses):
                try:
                    house_info = {
                        'index': int(i),
                        'address': str(self.original_data.iloc[i]['ADDRESS']) if 'ADDRESS' in self.original_data.columns else f"House {i}",
                        'bedrooms': int(self.original_data.iloc[i]['BEDROOMS']) if 'BEDROOMS' in self.original_data.columns and pd.notna(self.original_data.iloc[i]['BEDROOMS']) else 0,
                        'suburb': str(self.original_data.iloc[i]['SUBURB']) if 'SUBURB' in self.original_data.columns else 'Unknown',
                        'actual_price': float(self.target.iloc[i]) if pd.notna(self.target.iloc[i]) else 0.0
                    }
                    houses.append(house_info)
                except Exception as e:
                    print(f"Error processing house {i}: {str(e)}")
                    continue
            
            print(f"Successfully processed {len(houses)} houses")
            return houses
        except Exception as e:
            print(f"Error in get_houses_list: {str(e)}")
            raise e

    def select_data_by_index(self, idx):
        """Select data by index for web app use"""
        self.load_and_prepare_data()
        if 0 <= idx < len(self.df):
            selected_row = self.df.iloc[idx]
            actual_price = self.target.iloc[idx]
            return selected_row.values.reshape(1, -1), actual_price
        else:
            raise ValueError(f"Index {idx} is out of range. Valid range: 0 to {len(self.df)-1}")

    def select_data(self):
        """Interactive data selection for terminal use"""
        self.load_and_prepare_data()
        print(f"Loaded and processed {len(self.df)} rows.")
        print(f"Select a row index (0 to {len(self.df)-1}):")
        while True:
            try:
                idx = int(input("Enter row index: "))
                if 0 <= idx < len(self.df):
                    selected_row = self.df.iloc[idx]
                    actual_price = self.target.iloc[idx]
                    print("Selected row:")
                    print(selected_row)
                    print(f"Actual price: ${actual_price:,.0f}")
                    return selected_row.values.reshape(1, -1), actual_price
                else:
                    print(f"Please enter a number between 0 and {len(self.df)-1}.")
            except ValueError:
                print("Please enter a valid integer.")

    def predict_custom_data(self, custom_data):
        """Make prediction for custom house data using the same pipeline as training"""
        try:
            print(f"Input data: {custom_data}")
            
            # Create a DataFrame with the custom data in the same format as original CSV
            import datetime
            current_year = datetime.datetime.now().year
            
            # Create a row that mimics the original CSV structure
            custom_df = pd.DataFrame([{
                'BEDROOMS': custom_data['bedrooms'],
                'BUILD_YEAR': custom_data['build_year'],
                'FLOOR_AREA': custom_data['floor_area'],
                'LAND_AREA': custom_data['land_area'],
                'BATHROOMS': custom_data['bathrooms'],
                'GARAGE': custom_data['garage'],
                'NEAREST_STN_DIST': custom_data['nearest_stn_dist'],
                'NEAREST_SCH_DIST': custom_data['nearest_sch_dist'],
                'CBD_DIST': custom_data['cbd_dist'],
                'SUBURB': 'AVERAGE_SUBURB',  # Default suburb
                'NEAREST_STN': 'AVERAGE_STATION',  # Default station
                'NEAREST_SCH': 'AVERAGE_SCHOOL',  # Default school
                'PRICE': 500000,  # Dummy price (will be ignored)
                'ADDRESS': 'Custom Property',
                'DATE_SOLD': '2024-01-01',
                'LATITUDE': -31.9505,  # Perth coordinates
                'LONGITUDE': 115.8605
            }])
            
            print(f"Created custom DataFrame with shape: {custom_df.shape}")
            
            # Create a new DataProcessor instance for this custom data
            custom_processor = DataProcessor(custom_df)
            
            # Process the data through the same pipeline
            custom_processor.load_data()
            custom_processor.clean_data()
            custom_processor.add_features()
            
            # Get the features before normalization to use the original scaler
            features_before_norm = custom_processor.features
            print(f"Features before normalization: {features_before_norm.shape}")
            
            # Load the original scaler and feature structure
            if self.df is None:
                self.load_and_prepare_data()
            
            # Create a feature vector that matches the training data structure
            final_features = np.zeros(len(self.df.columns))
            
            # Map the features that exist in both datasets
            for col in features_before_norm.columns:
                if col in self.df.columns:
                    # Get the value and normalize it using training data statistics
                    value = features_before_norm[col].iloc[0]
                    col_min = self.df[col].min()
                    col_max = self.df[col].max()
                    
                    if col_max != col_min:
                        normalized_value = (value - col_min) / (col_max - col_min)
                    else:
                        normalized_value = 0
                    
                    feature_idx = self.df.columns.get_loc(col)
                    final_features[feature_idx] = normalized_value
                    print(f"Mapped {col}: {value} -> {normalized_value}")
            
            print(f"Final feature vector shape: {final_features.shape}")
            print(f"Non-zero features: {np.count_nonzero(final_features)}")
            
            # Ensure model is loaded
            if self.model is None:
                self.model = jlb.load(self.model_file)
            
            # Make prediction
            prediction = self.model.predict(final_features.reshape(1, -1))[0]
            print(f"Raw prediction: {prediction}")
            
            return prediction
            
        except Exception as e:
            print(f"Error in predict_custom_data: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def predict_data(self, selected_row, actual_price):
        """Make prediction and calculate differences"""
        if self.model is None:
            self.model = jlb.load(self.model_file)
        predicted_price = self.model.predict(selected_row)[0]
        diff = predicted_price - actual_price
        percent_diff = (diff / actual_price) * 100
        return predicted_price, diff, percent_diff

if __name__ == "__main__":
    predictor = Predictor()
    selected_row, actual_price = predictor.select_data()
    predictor.predict_data(selected_row, actual_price)