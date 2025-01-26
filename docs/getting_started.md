# Getting Started with Titanic Survival Analysis

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)
- Git

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/fahad0samara/data-analysis-titanic.git
   cd data-analysis-titanic
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset**
   ```bash
   python src/download_data.py
   ```

5. **Train the Model**
   ```bash
   python src/train_model.py
   ```

6. **Run the Dashboard**
   ```bash
   streamlit run src/dashboard.py
   ```

## 🎯 First Steps

1. **Explore the Dashboard**
   - Open http://localhost:8501 in your browser
   - Try different visualizations
   - Use the prediction tool

2. **Understanding the Data**
   - View passenger statistics
   - Analyze survival patterns
   - Explore feature correlations

3. **Making Predictions**
   - Enter passenger details
   - Get survival probability
   - Understand feature importance

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
MODEL_PATH=models/artifacts/best_model.joblib
DATA_PATH=data/raw/titanic.csv
DEBUG=True
```

### Model Parameters
Adjust model parameters in `src/config/config.py`:
```python
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Not Found**
   ```bash
   python src/download_data.py
   ```

3. **Model Training Errors**
   - Check available memory
   - Verify data integrity
   - Review model parameters

### Getting Help
- Open an issue on GitHub
- Check existing documentation
- Join our community discussions
