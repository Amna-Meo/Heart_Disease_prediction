# Heart Disease Prediction

## Project Overview
Predict the risk of heart disease using the UCI Heart Disease dataset with classification models.

## Dataset
- **Source**: UCI Heart Disease (Cleveland dataset)
- **Size**: 303 samples, 14 features
- **Target**: Binary classification (0 = No disease, 1 = Disease)
- **Clean Data**: 297 samples after removing missing values

## Project Structure
```
Heart Disease Prediction/
├── download_data.py          # Download dataset from UCI
├── data_cleaning.py          # Data cleaning and preprocessing
├── eda.py                    # Exploratory data analysis
├── train_models.py           # Model training and evaluation
├── requirements.txt          # Python dependencies
├── heart_disease.csv         # Raw dataset
├── heart_disease_clean.csv   # Cleaned dataset
├── eda_visualizations.png    # EDA plots
└── model_evaluation.png      # Model performance plots
```

## Key Features
The most important features for predicting heart disease:
1. **thal** (Thalassemia) - Most important in both models
2. **ca** (Number of major vessels) - Strong predictor
3. **cp** (Chest pain type) - Significant indicator
4. **oldpeak** (ST depression) - Important cardiac measure
5. **sex** - Gender correlation with disease

## Model Performance

### Logistic Regression
- **Test Accuracy**: 83.33%
- **ROC AUC Score**: 0.9498
- **Precision**: 85% (Disease), 82% (No Disease)
- **Recall**: 79% (Disease), 88% (No Disease)

### Decision Tree
- **Test Accuracy**: 70.00%
- **ROC AUC Score**: 0.7450
- **Precision**: 75% (Disease), 68% (No Disease)
- **Recall**: 54% (Disease), 84% (No Disease)

## Results Summary
- **Best Model**: Logistic Regression outperforms Decision Tree
- **ROC AUC**: 0.95 indicates excellent discrimination ability
- **Balanced Performance**: Good precision and recall for both classes
- **Key Insight**: Thalassemia and number of major vessels are the strongest predictors

## Installation & Usage

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Pipeline
```bash
# Download data
python download_data.py

# Clean data
python data_cleaning.py

# Perform EDA
python eda.py

# Train models
python train_models.py
```

## Visualizations
- **eda_visualizations.png**: Distribution plots, correlations, and feature analysis
- **model_evaluation.png**: Confusion matrices, ROC curves, feature importance, and model comparison

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Medical Context
This model identifies patients at risk of heart disease based on clinical measurements. The high ROC AUC score (0.95) suggests the model can effectively distinguish between patients with and without heart disease, making it potentially useful for clinical screening.
