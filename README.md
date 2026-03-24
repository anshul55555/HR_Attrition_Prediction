# HR Attrition Prediction

## Overview
This project predicts whether an employee is likely to leave the company using classification models. It covers the full machine learning workflow from data understanding to model evaluation and business insights.

## Objective
The goal is to identify employees at risk of attrition so organizations can improve retention strategies and make better HR decisions.

## Workflow
- Data loading and structure checking
- Data cleaning
- Exploratory Data Analysis (EDA)
- Outlier handling using IQR capping
- Encoding and scaling
- Train-test split
- Class imbalance handling with SMOTE
- Model training and comparison
- Cross-validation
- Hyperparameter tuning
- Best model selection
- Business interpretation

## Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

## Best Model
**Tuned Logistic Regression** was selected as the best business-balanced model.

### Test Performance
- Accuracy: **0.786**
- Precision: **0.400**
- Recall: **0.681**
- F1 Score: **0.504**
- ROC-AUC: **0.809**

## Why This Model Was Chosen
For attrition prediction, recall and F1 score are important because missing employees who are likely to leave can negatively affect the business. The tuned Logistic Regression model gave the best balance of detection performance and ranking quality.

## Key Insights
Employees showed higher attrition risk in patterns related to:
- Overtime
- Frequent business travel
- Delayed promotion
- Single marital status
- Higher number of previous companies worked

Lower attrition was generally associated with:
- Better satisfaction measures
- More work experience
- Higher age
- Stable job conditions

## Business Recommendations
- Reduce overtime pressure
- Monitor high-travel employees
- Improve promotion and career growth visibility
- Focus retention efforts on high-risk roles
- Strengthen work-life balance initiatives
- Build an early-warning system for attrition risk

## Project Output
The project delivers a complete classification pipeline with analysis, model comparison, evaluation, and interpretable business recommendations.
