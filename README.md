# HR Attrition Prediction

## Overview

This project predicts whether an employee is likely to leave a company using machine learning classification models. It follows a complete machine learning workflow, starting from data understanding and preprocessing to model evaluation, hyperparameter tuning, and business interpretation.

The main purpose of this project is to help organizations identify employees who are at risk of attrition and support better employee retention strategies.

---

## Objective

The objective of this project is to build a reliable classification model that can predict employee attrition based on HR-related features.

By identifying employees who are more likely to leave, organizations can:

- Take early preventive action
- Improve employee satisfaction
- Reduce employee turnover
- Support better HR decision-making
- Build data-driven retention strategies

---

## Workflow

The project follows the complete machine learning pipeline:

1. Data loading and structure checking
2. Data cleaning
3. Exploratory Data Analysis
4. Outlier handling using IQR capping
5. Encoding categorical variables
6. Feature scaling
7. Train-test split
8. Class imbalance handling using SMOTE
9. Model training and comparison
10. Cross-validation
11. Hyperparameter tuning
12. Best model selection
13. Business interpretation and recommendations

---

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors

---

## Best Model

The **Tuned Logistic Regression** model was selected as the best business-balanced model.

Although other models were also evaluated, Logistic Regression provided the best balance between interpretability and performance. This is important in HR analytics because the model should not only predict attrition but also help explain the reasons behind employee attrition risk.

---

## Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.786 |
| Precision | 0.400 |
| Recall | 0.681 |
| F1 Score | 0.504 |
| ROC-AUC | 0.809 |

---

## Why This Model Was Chosen

For employee attrition prediction, **recall** and **F1 score** are very important metrics.

Recall is important because failing to identify employees who are likely to leave can negatively affect the organization. A model with good recall helps HR teams detect more at-risk employees early.

The tuned Logistic Regression model was selected because it gave a strong balance between:

- Detecting employees likely to leave
- Maintaining reasonable prediction accuracy
- Providing interpretable results
- Supporting business decision-making
- Ranking attrition risk effectively using ROC-AUC

---

## Key Insights

Employees showed a higher risk of attrition in patterns related to:

- Overtime
- Frequent business travel
- Delayed promotion
- Single marital status
- Higher number of previous companies worked

Lower attrition risk was generally associated with:

- Better satisfaction levels
- More work experience
- Higher age
- Stable job conditions
- Better work-life balance

---

## Business Recommendations

- Reduce overtime pressure
- Monitor employees with frequent business travel
- Improve promotion transparency and career growth opportunities
- Focus retention efforts on high-risk job roles
- Strengthen work-life balance initiatives
- Improve employee satisfaction through regular feedback
- Build an early-warning system to identify employees at risk of leaving

---

## Project Output

This project delivers a complete HR attrition classification pipeline that includes:

- Data preprocessing
- Exploratory Data Analysis
- Outlier handling
- Feature encoding and scaling
- Class imbalance treatment using SMOTE
- Multiple model training and comparison
- Cross-validation
- Hyperparameter tuning
- Best model selection
- Business insights and recommendations

The final output is a machine learning model that can support HR teams in predicting employee attrition and making better retention decisions.

---

## Conclusion

The HR Attrition Prediction project demonstrates how machine learning can be used to solve real-world HR problems. By identifying employees who are likely to leave, companies can take proactive steps to improve employee satisfaction, reduce turnover, and make more effective workforce decisions.
