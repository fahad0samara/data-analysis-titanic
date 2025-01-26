# API Reference

## 🔌 REST API Endpoints

### Prediction API

#### Make a Prediction
```http
POST /api/v1/predict
```

**Request Body:**
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

**Response:**
```json
{
  "passenger_id": 1,
  "survival_probability": 0.75,
  "predicted_survival": true,
  "prediction_timestamp": "2025-01-26T10:55:12+02:00",
  "model_version": "v1.0.0",
  "feature_importance": {
    "age": 0.15,
    "fare": 0.25,
    "pclass": 0.3
  }
}
```

#### Get Model Metrics
```http
GET /api/v1/model/metrics
```

**Response:**
```json
{
  "model_version": "v1.0.0",
  "accuracy": 0.85,
  "precision": 0.83,
  "recall": 0.87,
  "f1_score": 0.85,
  "roc_auc": 0.89,
  "training_timestamp": "2025-01-26T10:55:12+02:00"
}
```

## 🔧 Python API

### Model Training

```python
from src.application.services import ModelService
from src.domain.entities import Passenger

# Initialize service
model_service = ModelService()

# Train model
metrics = model_service.train(training_data)
print(f"Model accuracy: {metrics.accuracy}")
```

### Making Predictions

```python
# Create passenger instance
passenger = Passenger(
    passenger_id=1,
    pclass=1,
    name="John Doe",
    sex="male",
    age=30,
    sibsp=1,
    parch=0,
    ticket="TEST123",
    fare=100,
    cabin=None,
    embarked="S"
)

# Make prediction
prediction = model_service.predict(passenger)
print(f"Survival probability: {prediction.survival_probability}")
```

### Data Processing

```python
from src.utils.data_utils import (
    load_and_preprocess_data,
    create_age_groups,
    calculate_survival_statistics
)

# Load and preprocess data
df = load_and_preprocess_data("data/raw/titanic.csv")

# Create age groups
df = create_age_groups(df)

# Calculate statistics
stats = calculate_survival_statistics(df)
```

### Visualization

```python
from src.utils.visualization_utils import (
    create_survival_distribution,
    create_age_distribution,
    create_correlation_heatmap
)

# Create visualizations
survival_plot = create_survival_distribution(df, "Pclass", "Survival by Class")
age_plot = create_age_distribution(df)
correlation_plot = create_correlation_heatmap(df, ["Age", "Fare", "Pclass"])
```

## 🔒 Authentication

The API uses token-based authentication:

```http
Authorization: Bearer <your_token>
```

To get an API token:
1. Register at `/api/v1/auth/register`
2. Login at `/api/v1/auth/login`
3. Use the returned token in the Authorization header

## 📊 Rate Limiting

- 100 requests per minute for authenticated users
- 10 requests per minute for unauthenticated users
- 1000 requests per day per API key

## 🚨 Error Handling

### Error Response Format
```json
{
  "error": "Invalid passenger data",
  "error_code": "VALIDATION_ERROR",
  "timestamp": "2025-01-26T10:55:12+02:00"
}
```

### Common Error Codes
- `VALIDATION_ERROR`: Invalid input data
- `MODEL_ERROR`: Model prediction failed
- `AUTH_ERROR`: Authentication failed
- `RATE_LIMIT_ERROR`: Too many requests
- `SERVER_ERROR`: Internal server error
