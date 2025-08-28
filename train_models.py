#!/usr/bin/env python3
"""
CO2 Emission Prediction - Model Training Script
This script trains multiple machine learning models for CO2 emission prediction
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CO2EmissionPredictor:
    def __init__(self, data_path='FuelConsumptionCo2.csv'):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Select relevant features
        features = [
            'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 
            'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB'
        ]
        target = 'CO2EMISSIONS'
        
        # Check if all required columns exist
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            # Try alternative column names
            if 'ENGINESIZE' not in df.columns and 'Engine Size' in df.columns:
                features = ['Engine Size', 'Cylinders', 'City Fuel Consumption', 
                          'Highway Fuel Consumption', 'Combined Fuel Consumption']
                target = 'CO2 Emissions'
        
        # Remove rows with missing values
        df_clean = df[features + [target]].dropna()
        print(f"Clean dataset shape: {df_clean.shape}")
        
        # Split features and target
        X = df_clean[features]
        y = df_clean[target]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        self.feature_names = features
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        return X_scaled, y
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\nTraining models...")
        
        # Define models to train
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, gamma='scale')
        }
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store model and metrics
            self.models[name] = {
                'model': model,
                'metrics': {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'CV_Mean': cv_mean,
                    'CV_Std': cv_std
                }
            }
            
            print(f"  {name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, CV: {cv_mean:.4f} ± {cv_std:.4f}")
            
            # Update best model
            if cv_mean > self.best_score:
                self.best_score = cv_mean
                self.best_model = name
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best model"""
        print(f"\nPerforming hyperparameter tuning for {self.best_model}...")
        
        if 'Random Forest' in self.best_model:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestRegressor(random_state=42)
            
        elif 'Gradient Boosting' in self.best_model:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = GradientBoostingRegressor(random_state=42)
            
        else:
            print("Hyperparameter tuning not implemented for this model type")
            return
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        # Update best model
        self.models[self.best_model]['model'] = grid_search.best_estimator_
        self.models[self.best_model]['best_params'] = grid_search.best_params_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\nModel Evaluation Summary:")
        print("=" * 80)
        print(f"{'Model':<20} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'CV Score':<12}")
        print("=" * 80)
        
        for name, model_info in self.models.items():
            metrics = model_info['metrics']
            print(f"{name:<20} {metrics['R2']:<8.4f} {metrics['RMSE']:<8.2f} "
                  f"{metrics['MAE']:<8.2f} {metrics['CV_Mean']:<8.4f} ± {metrics['CV_Std']:<8.4f}")
        
        print("=" * 80)
        print(f"Best Model: {self.best_model} (CV Score: {self.best_score:.4f})")
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        print(f"\nSaving models to {output_dir}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each model
        for name, model_info in self.models.items():
            filename = f"{output_dir}/{name.lower().replace(' ', '_')}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model_info['model'], f)
            print(f"  Saved {name} to {filename}")
        
        # Save scaler
        scaler_file = f"{output_dir}/scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  Saved scaler to {scaler_file}")
        
        # Save model comparison
        comparison_file = f"{output_dir}/model_comparison.txt"
        with open(comparison_file, 'w') as f:
            f.write("CO2 Emission Prediction - Model Comparison\n")
            f.write("=" * 50 + "\n\n")
            for name, model_info in self.models.items():
                metrics = model_info['metrics']
                f.write(f"{name}:\n")
                f.write(f"  R² Score: {metrics['R2']:.4f}\n")
                f.write(f"  RMSE: {metrics['RMSE']:.2f}\n")
                f.write(f"  MAE: {metrics['MAE']:.2f}\n")
                f.write(f"  CV Score: {metrics['CV_Mean']:.4f} ± {metrics['CV_Std']:.4f}\n")
                if 'best_params' in model_info:
                    f.write(f"  Best Parameters: {model_info['best_params']}\n")
                f.write("\n")
        
        print(f"  Saved model comparison to {comparison_file}")
    
    def create_visualizations(self, output_dir='models'):
        """Create visualization plots"""
        print("\nCreating visualizations...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Model performance comparison
        plt.figure(figsize=(12, 8))
        
        # R² scores
        plt.subplot(2, 2, 1)
        names = list(self.models.keys())
        r2_scores = [self.models[name]['metrics']['R2'] for name in names]
        plt.bar(names, r2_scores, color='skyblue')
        plt.title('Model R² Scores')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # RMSE scores
        plt.subplot(2, 2, 2)
        rmse_scores = [self.models[name]['metrics']['RMSE'] for name in names]
        plt.bar(names, rmse_scores, color='lightcoral')
        plt.title('Model RMSE Scores')
        plt.ylabel('RMSE (g/km)')
        plt.xticks(rotation=45)
        
        # Cross-validation scores
        plt.subplot(2, 2, 3)
        cv_means = [self.models[name]['metrics']['CV_Mean'] for name in names]
        cv_stds = [self.models[name]['metrics']['CV_Std'] for name in names]
        plt.bar(names, cv_means, yerr=cv_stds, color='lightgreen', capsize=5)
        plt.title('Cross-Validation Scores')
        plt.ylabel('CV Score')
        plt.xticks(rotation=45)
        
        # Feature importance (for Random Forest)
        plt.subplot(2, 2, 4)
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']['model']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.bar(range(len(importances)), importances[indices])
            plt.title('Feature Importance (Random Forest)')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(range(len(importances)), 
                       [self.feature_names[i] for i in indices], rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_performance.png', dpi=300, bbox_inches='tight')
        print(f"  Saved visualization to {output_dir}/model_performance.png")
        
        # Actual vs Predicted plot for best model
        best_model_info = self.models[self.best_model]
        y_pred = best_model_info['model'].predict(self.X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual CO2 Emissions (g/km)')
        plt.ylabel('Predicted CO2 Emissions (g/km)')
        plt.title(f'Actual vs Predicted CO2 Emissions ({self.best_model})')
        plt.grid(True, alpha=0.3)
        
        # Add R² text
        r2 = best_model_info['metrics']['R2']
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        print(f"  Saved actual vs predicted plot to {output_dir}/actual_vs_predicted.png")

def main():
    """Main function to run the complete training pipeline"""
    print("CO2 Emission Prediction - Model Training Pipeline")
    print("=" * 60)
    
    # Initialize predictor
    predictor = CO2EmissionPredictor()
    
    try:
        # Load and preprocess data
        predictor.load_and_preprocess_data()
        
        # Train models
        predictor.train_models()
        
        # Hyperparameter tuning for best model
        predictor.hyperparameter_tuning()
        
        # Evaluate models
        predictor.evaluate_models()
        
        # Save models
        predictor.save_models()
        
        # Create visualizations
        predictor.create_visualizations()
        
        print("\nTraining pipeline completed successfully!")
        print(f"Best model: {predictor.best_model}")
        print(f"Best CV score: {predictor.best_score:.4f}")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
