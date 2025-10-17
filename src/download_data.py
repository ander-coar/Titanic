"""
Script to download the Titanic dataset
"""
import pandas as pd
import os

def download_titanic_data():
    """Create Titanic dataset with sample data"""
    
    # Create sample Titanic data
    # This is a subset of the famous Titanic dataset
    data = {
        'survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1,
                    1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1,
                    0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0,
                    0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                    1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
        'pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3,
                  2, 2, 3, 1, 3, 1, 3, 1, 3, 3, 1, 1, 3, 2, 1, 3, 1, 3, 3, 3,
                  3, 2, 3, 2, 3, 3, 2, 1, 3, 3, 3, 2, 1, 1, 2, 1, 2, 3, 1, 2,
                  3, 1, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 1, 3, 2, 3,
                  3, 2, 1, 3, 1, 3, 1, 3, 2, 3, 1, 1, 2, 1, 2, 3, 3, 2, 1, 3],
        'sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female',
                'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male', 'female',
                'female', 'male', 'female', 'female', 'male', 'female', 'male', 'male', 'female', 'male',
                'male', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'female',
                'male', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'male', 'male',
                'male', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'male',
                'male', 'male', 'male', 'female', 'male', 'male', 'female', 'female', 'male', 'female',
                'male', 'male', 'male', 'male', 'male', 'female', 'male', 'male', 'male', 'female',
                'female', 'male', 'female', 'female', 'male', 'male', 'female', 'male', 'male', 'male',
                'male', 'male', 'male', 'male', 'female', 'female', 'male', 'female', 'female', 'male'],
        'age': [22, 38, 26, 35, 35, 27, 54, 2, 27, 14, 4, 58, 20, 39, 14, 55, 2, 31, 28, 8,
               38, 19, 33, 40, 66, 28, 42, 21, 18, 14, 40, 27, 3, 19, 18, 7, 21, 49, 29, 65,
               28.5, 5, 11, 22, 38, 45, 4, 29, 19, 17, 26, 32, 16, 21, 26, 32, 25, 0.83, 30, 22,
               29, 28, 17, 33, 16, 18, 34, 29, 22, 2, 21, 26, 18.5, 32, 16, 25, 58, 17, 14, 50,
               24, 45, 33, 20, 47, 29, 25, 23, 19, 37, 16, 24, 22, 24, 19, 27, 9, 36.5, 42, 51],
        'sibsp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1, 1, 0, 0, 1, 0, 0, 4, 1, 0, 3,
                 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 4, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0,
                 0, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 8, 0, 0, 0, 0],
        'parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 5, 0, 0, 1, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 1, 0, 0, 0, 0,
                 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
        'fare': [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 51.86, 21.07, 11.13, 30.07,
                16.70, 26.55, 8.05, 31.27, 7.85, 16.00, 29.12, 13.00, 13.00, 21.07,
                21.00, 7.25, 7.63, 27.72, 10.50, 82.17, 52.00, 7.23, 5.00, 7.75,
                83.16, 26.00, 19.25, 7.23, 60.00, 22.02, 7.25, 7.75, 7.90, 26.00,
                10.46, 82.17, 52.00, 7.23, 5.00, 7.75, 14.45, 61.97, 7.75, 20.57,
                7.92, 7.75, 7.88, 24.15, 13.00, 52.55, 5.00, 21.68, 55.00, 16.00,
                7.75, 29.70, 13.00, 77.96, 7.90, 7.75, 7.73, 9.50, 7.78, 15.85,
                7.75, 21.00, 7.73, 39.69, 7.78, 26.00, 7.88, 26.00, 7.73, 8.66,
                10.50, 26.00, 26.55, 16.10, 26.00, 7.75, 13.00, 8.05, 26.00, 7.73,
                14.50, 7.79, 21.67, 5.00, 21.68, 7.23, 39.69, 6.95, 31.27, 7.05],
        'embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C',
                    'S', 'S', 'S', 'S', 'S', 'S', 'C', 'S', 'S', 'S',
                    'S', 'S', 'S', 'C', 'S', 'C', 'S', 'S', 'Q', 'S',
                    'C', 'S', 'S', 'S', 'S', 'S', 'Q', 'S', 'Q', 'C',
                    'S', 'C', 'S', 'S', 'Q', 'S', 'C', 'C', 'Q', 'S',
                    'S', 'Q', 'S', 'S', 'S', 'C', 'Q', 'S', 'C', 'S',
                    'Q', 'C', 'S', 'S', 'Q', 'S', 'S', 'C', 'Q', 'S',
                    'Q', 'S', 'S', 'C', 'Q', 'C', 'C', 'Q', 'C', 'C',
                    'S', 'C', 'S', 'S', 'C', 'Q', 'C', 'S', 'C', 'S',
                    'S', 'S', 'C', 'Q', 'S', 'S', 'C', 'S', 'S', 'S']
    }
    
    # Create DataFrame
    titanic = pd.DataFrame(data)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    titanic.to_csv('data/titanic.csv', index=False)
    print(f"Dataset created successfully! Shape: {titanic.shape}")
    print(f"Saved to: data/titanic.csv")
    print(f"\nFirst few rows:")
    print(titanic.head())
    
    return titanic

if __name__ == "__main__":
    download_titanic_data()
