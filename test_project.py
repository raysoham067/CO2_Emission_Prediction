#!/usr/bin/env python3
"""
Test script for CO2 Emission Prediction project
This script tests the main components to ensure everything works correctly
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle

def test_data_loading():
    """Test if the dataset can be loaded correctly"""
    print("Testing data loading...")
    
    try:
        # Check if dataset exists
        if not os.path.exists('FuelConsumptionCo2.csv'):
            print("❌ Dataset not found!")
            return False
        
        # Load dataset
        df = pd.read_csv('FuelConsumptionCo2.csv')
        print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        
        # Check required columns
        required_cols = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 
                        'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
            # Try alternative column names
            alt_cols = ['Engine Size', 'Cylinders', 'City Fuel Consumption', 
                       'Highway Fuel Consumption', 'Combined Fuel Consumption', 'CO2 Emissions']
            if all(col in df.columns for col in alt_cols):
                print("✅ Alternative column names found")
                return True
            return False
        else:
            print("✅ All required columns found")
            return True
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_model_training():
    """Test if models can be trained"""
    print("\nTesting model training...")
    
    try:
        # Load dataset
        df = pd.read_csv('FuelConsumptionCo2.csv')
        
        # Try to find the right column names
        if 'ENGINESIZE' in df.columns:
            features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 
                      'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB']
            target = 'CO2EMISSIONS'
        elif 'Engine Size' in df.columns:
            features = ['Engine Size', 'Cylinders', 'City Fuel Consumption', 
                      'Highway Fuel Consumption', 'Combined Fuel Consumption']
            target = 'CO2 Emissions'
        else:
            print("❌ Could not find appropriate column names")
            return False
        
        # Clean data
        df_clean = df[features + [target]].dropna()
        if len(df_clean) == 0:
            print("❌ No clean data available after removing missing values")
            return False
        
        print(f"✅ Clean data shape: {df_clean.shape}")
        
        # Prepare features and target
        X = df_clean[features].values
        y = df_clean[target].values
        
        # Train a simple model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make a prediction
        sample_input = np.array([[2.0, 4, 8.5, 6.2, 7.5]])
        prediction = model.predict(sample_input)[0]
        
        print(f"✅ Model trained successfully! Sample prediction: {prediction:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def test_flask_imports():
    """Test if Flask and required packages can be imported"""
    print("\nTesting Flask imports...")
    
    try:
        from flask import Flask, render_template, request, jsonify
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        
        print("✅ All required packages imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        return False

def test_templates():
    """Test if HTML templates exist"""
    print("\nTesting HTML templates...")
    
    required_templates = ['templates/index.html', 'templates/result.html']
    
    for template in required_templates:
        if os.path.exists(template):
            print(f"✅ {template} found")
        else:
            print(f"❌ {template} not found")
            return False
    
    return True

def test_requirements():
    """Test if requirements.txt exists"""
    print("\nTesting requirements.txt...")
    
    if os.path.exists('requirements.txt'):
        print("✅ requirements.txt found")
        return True
    else:
        print("❌ requirements.txt not found")
        return False

def main():
    """Run all tests"""
    print("🧪 CO2 Emission Prediction Project - Test Suite")
    print("=" * 50)
    
    tests = [
        test_requirements,
        test_data_loading,
        test_model_training,
        test_flask_imports,
        test_templates
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your project is ready to run.")
        print("\n🚀 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train models: python train_models.py")
        print("3. Run web app: python main.py")
        print("4. Open browser: http://localhost:5000")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the project.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
