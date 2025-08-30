# Shift ML Prediction Model

## Project Goal

This project develops machine learning models to predict customer lifetime value (LTV) and revenue at Day 180 using only the first 3 days of user behavioral data (Day 1 to Day 3).

## Business Objective

Create an accurate early-stage prediction model that can:
- Identify high-value users within their first 3 days
- Predict 6-month revenue outcomes for business planning
- Enable early intervention strategies for user retention and monetization

## Project Structure

```
shift_ml_prediction_model/
└── ben_work/
    ├── d1-d180-prediction-model_exclude_d1_uninstalls_v6_hybrid_refined.ipynb
    ├── d1-d180-prediction-model_v8_hybrid_revenue_events_only_d1_d3.ipynb
    └── README.md
```

## Approach

The models use a hybrid classification-regression approach:
1. **Classification**: Predict if a user will generate revenue (yes/no)
2. **Regression**: For revenue-generating users, predict the amount

This two-stage approach handles the challenge of zero-inflated revenue data where many users generate no revenue.

## Key Challenge

Predicting 180-day outcomes from just 3 days of data represents an extreme extrapolation challenge (60x time horizon). The models aim to identify early behavioral signals that correlate with long-term value.

## Expected Outcomes

The models should achieve:
- High accuracy in identifying revenue vs non-revenue generating users
- Reasonable revenue amount predictions for business decision-making
- Actionable insights about which early behaviors predict long-term value

## Technical Stack

- **Languages**: Python
- **ML Frameworks**: XGBoost, CatBoost, LightGBM, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Cloud Services**: AWS S3 for data storage
- **Development**: Jupyter Notebooks

## Getting Started

See `/ben_work/README.md` for detailed information about the models, validation methods, and performance metrics.

## Note on Validation

The models require evaluation against actual Day 180 revenue data to verify accuracy. Performance metrics shown in notebook outputs represent results from previous runs with data that is not included in this repository.