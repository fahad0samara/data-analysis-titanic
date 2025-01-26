import pandas as pd
import os

def download_titanic_data():
    # URL for the Titanic dataset
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    try:
        # Download and save the dataset
        df = pd.read_csv(url)
        df.to_csv('data/raw/titanic.csv', index=False)
        print("Successfully downloaded Titanic dataset to data/raw/titanic.csv")
        print(f"Dataset shape: {df.shape}")
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")

if __name__ == "__main__":
    download_titanic_data()
