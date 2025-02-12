# Student Performance Prediction

## Overview
This project aims to predict students' performance in mathematics based on various features such as gender, parental education, lunch type, test preparation course, and reading/writing scores. The project follows a machine learning pipeline that includes data ingestion, transformation, and model training using multiple regression models.

## Installation & Setup
### Prerequisites
Ensure you have Python 3.x installed. Install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Workflow
### 1. Data Ingestion (`data_ingestion.py`)
- Reads the dataset from `notebook/data/StudentsPerformance.csv`.
- Splits data into training and testing sets (80-20 split).
- Stores processed data in the `artifacts/` directory.

### 2. Data Transformation (`data_transformation.py`)
- Applies preprocessing using Scikit-learn Pipelines:
  - Imputation of missing values.
  - Standard scaling for numerical features.
  - One-hot encoding for categorical features.
- Saves the preprocessor object (`preprocessor.pkl`).

### 3. Model Training (`model_trainer.py`)
- Trains multiple regression models:
  - Random Forest Regressor
  - Decision Tree Regressor
  - Gradient Boosting
  - Linear Regression
  - K-Neighbors Regressor
  - XGBoost
  - CatBoost
  - AdaBoost
- Evaluates models based on R² score.
- Selects and saves the best model (`model.pkl`).

## Logging & Exception Handling
- Logs execution flow using `exec/logger.py`.
- Custom exception handling with `exec/exception.py`.

## Expected Output
- A trained model saved in `artifacts/model.pkl`.
- Logging information tracking progress and issues.
- Console output displaying the best model and its R² score.

## Future Improvements
- Hyperparameter tuning for better performance.
- Adding more features for enhanced predictions.
- Deploying the model using Flask or FastAPI.


