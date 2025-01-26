# Titanic Survival Analysis Project

An advanced machine learning project that analyzes the Titanic dataset and predicts passenger survival using a clean architecture approach.

## 🌟 Features

- **Interactive Dashboard**
  - Real-time survival predictions
  - Dynamic data filtering
  - Interactive visualizations
  - Statistical analysis
  - Passenger information display

- **Machine Learning Model**
  - Random Forest Classifier
  - 80% accuracy on test data
  - Feature importance analysis
  - Cross-validation
  - Model metrics tracking

- **Clean Architecture**
  - Domain-driven design
  - Clear separation of concerns
  - SOLID principles
  - Dependency injection
  - Interface-based design

## 🏗️ Project Structure

```
├── data/
│   ├── raw/             # Original Titanic dataset
│   ├── processed/       # Cleaned and transformed data
│   └── interim/         # Intermediate processing results
├── models/
│   ├── artifacts/       # Trained model files
│   └── metrics/         # Model performance metrics
├── notebooks/           # Jupyter notebooks for analysis
├── src/
│   ├── application/     # Application use cases
│   │   ├── __init__.py
│   │   └── services.py  # Core business logic
│   ├── config/         # Configuration settings
│   │   ├── __init__.py
│   │   └── config.py   # App configuration
│   ├── domain/         # Business entities and rules
│   │   ├── __init__.py
│   │   ├── entities.py
│   │   ├── interfaces.py
│   │   └── exceptions.py
│   ├── infrastructure/ # External interfaces
│   │   ├── __init__.py
│   │   ├── repositories.py
│   │   └── validators.py
│   ├── utils/          # Helper functions
│   │   ├── __init__.py
│   │   ├── data_utils.py
│   │   └── visualization_utils.py
│   ├── api.py          # FastAPI service
│   ├── dashboard.py    # Streamlit dashboard
│   └── train_model.py  # Model training script
├── tests/              # Test files
└── requirements.txt    # Project dependencies
```

## 🚀 Quick Start

1. **Clone the Repository**
   ```bash
   git clone  https://github.com/fahad0samara/data-analysis-titanic.git
   cd titanic-survival-analysis
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

3. **Download Data & Train Model**
   ```bash
   python src/download_data.py
   python src/train_model.py
   ```

4. **Run the Dashboard**
   ```bash
   streamlit run src/dashboard.py
   ```

## 📊 Dashboard Guide

### Data Overview
- View total passenger count and survival rate
- Filter by passenger class (1st, 2nd, 3rd)
- Filter by gender (male/female)
- Analyze age and fare distributions

### Visualizations
- Age distribution by survival status
- Fare vs. Age scatter plot
- Survival rates by passenger class
- Family size impact on survival
- Correlation heatmap of features

### Prediction Tool
1. Enter passenger details:
   - Class (1-3)
   - Age
   - Gender
   - Fare
   - Family members
2. Click "Predict" for survival probability
3. View feature importance for the prediction

## 🧪 Model Performance

- Training Accuracy: 90%
- Testing Accuracy: 80%
- Key Features:
  - Passenger Class
  - Gender
  - Age
  - Fare
  - Family Size

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
- Type hints
- Docstrings
- Unit tests
- Clean architecture
- PEP 8 compliance

## 📝 API Documentation

### Endpoints
- `GET /api/v1/passengers`: List all passengers
- `GET /api/v1/passengers/{id}`: Get passenger details
- `POST /api/v1/predict`: Get survival prediction
- `GET /api/v1/model/metrics`: Get model performance metrics

### Request Example
```json
{
  "passenger": {
    "pclass": 1,
    "name": "John Doe",
    "sex": "male",
    "age": 30,
    "sibsp": 1,
    "parch": 0,
    "fare": 100,
    "embarked": "S"
  }
}
```

### Response Example
```json
{
  "passenger_id": 1,
  "survival_probability": 0.75,
  "predicted_survival": true,
  "prediction_timestamp": "2025-01-26T10:47:50+02:00",
  "model_version": "v1.0.0"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - feel free to use this project for learning, development, or production.

## 🙏 Acknowledgments

- Titanic dataset from Kaggle
- Streamlit for the dashboard framework
- FastAPI for the API service
- scikit-learn for machine learning
