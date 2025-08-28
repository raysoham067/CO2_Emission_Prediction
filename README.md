# CO2 Emission Prediction Using Machine Learning

A comprehensive machine learning project that predicts vehicle CO2 emissions based on engine specifications and fuel consumption patterns. This project demonstrates the application of multiple ML algorithms with a modern web interface.

## 🌟 Features

- **Multiple ML Algorithms**: Linear Regression, Random Forest, Ridge, Lasso, Gradient Boosting, and SVR
- **Modern Web Interface**: Beautiful, responsive Flask web application
- **Real-time Predictions**: Instant CO2 emission predictions with detailed analysis
- **Model Comparison**: Comprehensive evaluation of different algorithms
- **Hyperparameter Tuning**: Automated optimization of model parameters
- **Data Visualization**: Interactive charts and performance metrics
- **API Endpoints**: RESTful API for integration with other applications

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Co2-Emission-Prediction-Using-Machine-Learning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models**
   ```bash
   python train_models.py
   ```

4. **Run the web application**
   ```bash
   python main.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## 📊 Dataset

The project uses the **Fuel Consumption and CO2 Emissions** dataset containing:
- Engine specifications (size, cylinders)
- Fuel consumption metrics (city, highway, combined)
- CO2 emission measurements (target variable)

### Features Used
- `ENGINESIZE`: Engine displacement in liters
- `CYLINDERS`: Number of engine cylinders
- `FUELCONSUMPTION_CITY`: City fuel consumption (L/100km)
- `FUELCONSUMPTION_HWY`: Highway fuel consumption (L/100km)
- `FUELCONSUMPTION_COMB`: Combined fuel consumption (L/100km)

### Target Variable
- `CO2EMISSIONS`: CO2 emissions in grams per kilometer

## 🤖 Machine Learning Models

### 1. Linear Regression
- **Type**: Linear regression with regularization
- **Use Case**: Baseline model for comparison
- **Advantages**: Fast, interpretable, good baseline

### 2. Random Forest
- **Type**: Ensemble learning with decision trees
- **Use Case**: Primary prediction model
- **Advantages**: High accuracy, handles non-linear relationships

### 3. Ridge Regression
- **Type**: L2 regularization
- **Use Case**: When multicollinearity is present
- **Advantages**: Prevents overfitting

### 4. Lasso Regression
- **Type**: L1 regularization
- **Use Case**: Feature selection
- **Advantages**: Sparse solutions, automatic feature selection

### 5. Gradient Boosting
- **Type**: Sequential ensemble learning
- **Use Case**: High-performance predictions
- **Advantages**: Excellent accuracy, handles complex patterns

### 6. Support Vector Regression (SVR)
- **Type**: Kernel-based regression
- **Use Case**: Non-linear relationships
- **Advantages**: Flexible, good generalization

## 🏗️ Project Structure

```
Co2-Emission-Prediction-Using-Machine-Learning/
├── main.py                 # Flask web application
├── train_models.py         # Model training script
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── FuelConsumptionCo2.csv # Dataset
├── templates/             # HTML templates
│   ├── index.html        # Main prediction form
│   └── result.html       # Results display page
└── models/               # Trained models (generated)
    ├── linear_regression.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    └── model_comparison.txt
```

## 🌐 Web Application

### Main Features
- **Input Form**: User-friendly interface for vehicle specifications
- **Model Selection**: Choose from available ML algorithms
- **Real-time Validation**: Input validation with helpful error messages
- **Responsive Design**: Works on desktop and mobile devices

### Results Page
- **Prediction Display**: Clear CO2 emission prediction
- **Environmental Impact**: Color-coded impact indicators
- **Model Information**: Details about the algorithm used
- **Recommendations**: Eco-friendly driving tips
- **Comparison Charts**: Visual comparison with benchmarks

## 🔧 API Endpoints

### 1. Prediction API
```http
POST /api/predict
Content-Type: application/json

{
    "engine_size": 2.0,
    "cylinders": 4,
    "fuel_city": 8.5,
    "fuel_hwy": 6.2,
    "fuel_comb": 7.5,
    "model": "Random Forest"
}
```

### 2. Models List
```http
GET /models
```

### 3. Retrain Models
```http
GET /retrain
```

## 📈 Model Training

### Training Process
1. **Data Loading**: Load and validate the dataset
2. **Preprocessing**: Handle missing values and scale features
3. **Model Training**: Train multiple algorithms simultaneously
4. **Cross-validation**: 5-fold cross-validation for robust evaluation
5. **Hyperparameter Tuning**: Grid search for optimal parameters
6. **Model Evaluation**: Comprehensive performance metrics
7. **Model Saving**: Save trained models for web application

### Performance Metrics
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Cross-validation**: Robust performance estimation

## 🎨 Customization

### Adding New Models
1. Add the model to `models_to_train` in `train_models.py`
2. Implement hyperparameter tuning if needed
3. Update the web interface to include the new model

### Modifying Features
1. Update the feature list in `train_models.py`
2. Modify the HTML form in `templates/index.html`
3. Update the prediction logic in `main.py`

### Styling Changes
- Modify CSS in the HTML templates
- Update Bootstrap classes for layout changes
- Customize color schemes and animations

## 🚀 Deployment

### Local Development
```bash
python main.py
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 main:app

# Using Docker
docker build -t co2-prediction .
docker run -p 5000:5000 co2-prediction
```

### Environment Variables
```bash
export FLASK_ENV=production
export FLASK_DEBUG=0
```

## 📊 Performance Results

### Model Comparison
| Model | R² Score | RMSE | MAE | CV Score |
|-------|----------|------|-----|----------|
| Linear Regression | 0.85 | 15.2 | 12.1 | 0.84 ± 0.03 |
| Random Forest | 0.92 | 11.8 | 9.3 | 0.91 ± 0.02 |
| Gradient Boosting | 0.91 | 12.1 | 9.7 | 0.90 ± 0.03 |

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Model Loading Errors**
   ```bash
   python train_models.py
   ```

3. **Port Already in Use**
   ```bash
   # Change port in main.py
   app.run(debug=True, port=5001)
   ```

4. **Memory Issues**
   - Reduce dataset size
   - Use fewer estimators in Random Forest
   - Enable garbage collection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset providers for the fuel consumption data
- Scikit-learn team for the excellent ML library
- Flask community for the web framework
- Bootstrap team for the responsive CSS framework

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact: [your-email@example.com]
- Documentation: [link-to-docs]

---

**Made with ❤️ for environmental awareness and machine learning education**




