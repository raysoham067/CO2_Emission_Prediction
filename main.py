from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

app = Flask(__name__)

# Global variables for models
models = {}
current_model = None

def load_or_train_models():
    """Load existing models or train new ones if they don't exist"""
    global models, current_model
    
    if os.path.exists('models/linear_regression.pkl'):
        with open('models/linear_regression.pkl', 'rb') as f:
            models['Linear Regression'] = pickle.load(f)
    if os.path.exists('models/random_forest.pkl'):
        with open('models/random_forest.pkl', 'rb') as f:
            models['Random Forest'] = pickle.load(f)
    
    # If no models exist, train them
    if not models:
        train_models()
    
    # Set default model
    current_model = list(models.keys())[0] if models else None

def train_models():
    """Train machine learning models"""
    global models
    
    # Load and preprocess data
    try:
        df = pd.read_csv('FuelConsumptionCo2.csv')
        
        # Select relevant features for CO2 emission prediction
        features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB']
        target = 'CO2EMISSIONS'
        
        # Remove rows with missing values
        df_clean = df[features + [target]].dropna()
        
        X = df_clean[features]
        y = df_clean[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        with open('models/linear_regression.pkl', 'wb') as f:
            pickle.dump(lr_model, f)
        with open('models/random_forest.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        
        models['Linear Regression'] = lr_model
        models['Random Forest'] = rf_model
        
        print("Models trained and saved successfully!")
        
    except Exception as e:
        print(f"Error training models: {e}")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get form data
            engine_size = float(request.form['engine_size'])
            cylinders = int(request.form['cylinders'])
            fuel_city = float(request.form['fuel_city'])
            fuel_hwy = float(request.form['fuel_hwy'])
            fuel_comb = float(request.form['fuel_comb'])
            model_name = request.form.get('model', current_model)
            
            # Prepare input features
            input_features = np.array([[engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb]])
            
            # Make prediction
            if model_name in models:
                prediction = models[model_name].predict(input_features)[0]
                return render_template('result.html', 
                                    prediction=round(prediction, 2),
                                    model_name=model_name,
                                    input_data={
                                        'Engine Size': engine_size,
                                        'Cylinders': cylinders,
                                        'City Fuel Consumption': fuel_city,
                                        'Highway Fuel Consumption': fuel_hwy,
                                        'Combined Fuel Consumption': fuel_comb
                                    })
            else:
                return render_template('index.html', error="Selected model not found")
                
        except ValueError:
            return render_template('index.html', error="Please enter valid numeric values")
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")
    
    return render_template('index.html', models=list(models.keys()))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        engine_size = float(data['engine_size'])
        cylinders = int(data['cylinders'])
        fuel_city = float(data['fuel_city'])
        fuel_hwy = float(data['fuel_hwy'])
        fuel_comb = float(data['fuel_comb'])
        model_name = data.get('model', current_model)
        
        input_features = np.array([[engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb]])
        
        if model_name in models:
            prediction = models[model_name].predict(input_features)[0]
            return jsonify({
                'success': True,
                'prediction': round(prediction, 2),
                'model': model_name
            })
        else:
            return jsonify({'success': False, 'error': 'Model not found'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route("/models")
def list_models():
    """List available models"""
    return jsonify({
        'available_models': list(models.keys()),
        'current_model': current_model
    })

@app.route("/retrain")
def retrain():
    """Retrain all models"""
    try:
        train_models()
        return jsonify({'success': True, 'message': 'Models retrained successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Load or train models on startup
    load_or_train_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
