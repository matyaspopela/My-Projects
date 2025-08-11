from flask import Flask, render_template, request, jsonify
from predict import Predictor

app = Flask(__name__)

# Initialize predictor once
predictor = Predictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_houses')
def get_houses():
    """Return a list of houses with basic info for the dropdown"""
    try:
        print("Loading houses...")  # Debug print
        houses = predictor.get_houses_list()
        print(f"Loaded {len(houses)} houses")  # Debug print
        return jsonify(houses)
    except Exception as e:
        print(f"Error loading houses: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction for the selected house"""
    try:
        data = request.get_json()
        house_index = int(data['house_index'])
        print(f"Predicting for house index: {house_index}")  # Debug print
        
        # Use predictor to select data and make predictions
        selected_row, actual_price = predictor.select_data_by_index(house_index)
        predicted_price, diff, percent_diff = predictor.predict_data(selected_row, actual_price)
        
        result = {
            'predicted_price': float(predicted_price),
            'actual_price': float(actual_price),
            'difference': float(diff),
            'percentage_difference': float(percent_diff)
        }
        print(f"Prediction result: {result}")  # Debug print
        return jsonify(result)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

@app.route('/predict_custom', methods=['POST'])
def predict_custom():
    """Make a prediction for custom house data"""
    try:
        data = request.get_json()
        print(f"Custom prediction data: {data}")  # Debug print
        
        # Use predictor to make prediction with custom data
        predicted_price = predictor.predict_custom_data(data)
        
        result = {
            'predicted_price': float(predicted_price),
            'bedrooms': data['bedrooms'],
            'bathrooms': data['bathrooms'],
            'garage': data['garage'],
            'build_year': data['build_year'],
            'floor_area': data['floor_area'],
            'land_area': data['land_area'],
            'cbd_dist': data['cbd_dist'],
            'nearest_stn_dist': data['nearest_stn_dist'],
            'nearest_sch_dist': data['nearest_sch_dist']
        }
        print(f"Custom prediction result: {result}")  # Debug print
        return jsonify(result)
    except Exception as e:
        print(f"Error in custom prediction: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

