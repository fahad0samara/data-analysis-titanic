import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_model():
    # Load data
    df = pd.read_csv('data/raw/titanic.csv')
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features].copy()
    y = df['Survived']
    
    # Handle missing values
    X['Age'].fillna(X['Age'].median(), inplace=True)
    X['Fare'].fillna(X['Fare'].median(), inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    X['Sex'] = le.fit_transform(X['Sex'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_model.joblib')
    
    # Print performance
    print("\nModel Performance:")
    print(f"Training accuracy: {model.score(X_train, y_train):.2f}")
    print(f"Testing accuracy: {model.score(X_test, y_test):.2f}")

if __name__ == "__main__":
    train_model()
