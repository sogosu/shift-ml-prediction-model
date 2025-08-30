# Day 1-180 Revenue Prediction Models

This folder contains two Jupyter notebooks implementing machine learning models to predict customer lifetime value (LTV) and revenue at Day 180 using only the first 3 days of user data (Day 1 to Day 3).

## Notebooks

### 1. `d1-d180-prediction-model_exclude_d1_uninstalls_v6_hybrid_refined.ipynb`
- **Version 6**: Excludes users who uninstalled on Day 1 from training
- Implements a hybrid classification-regression approach
- Best reported performance: R² = 0.654, 83% predictions within ±10% accuracy

### 2. `d1-d180-prediction-model_v8_hybrid_revenue_events_only_d1_d3.ipynb`
- **Version 8**: Focuses on revenue-generating events from Days 1-3
- Also excludes D1 uninstalls and uses hybrid approach
- Similar performance to v6 model

## Model Architecture

Both notebooks implement a **two-stage hybrid approach**:

1. **Classification Stage**: Predicts whether a user will generate any revenue (binary classification)
   - Uses XGBoost Classifier
   - Handles zero-inflation in revenue data
   - Classification accuracy: 85-89%

2. **Regression Stage**: For users predicted to generate revenue, estimates the amount
   - Uses ensemble of XGBoost Regressor and MLP Neural Network
   - Trained on log-transformed revenue values
   - Regression-only R²: 0.25-0.27

## Data Pipeline

1. **Input Features**: User behavior data from Days 1-3
   - Mixpanel events
   - Ad operations revenue data
   - Query revenue data

2. **Feature Engineering**:
   - TF-IDF vectorization for text features
   - Target encoding for categorical variables
   - StandardScaler for numerical features
   - PCA for dimensionality reduction

3. **Data Sources**: AWS S3 buckets (requires credentials)

## Validation Methodology

- **Train/Test Split**: 15% holdout test set (random_state=42)
- **Metrics Evaluated**:
  - R² (R-squared) for variance explained
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - Classification metrics (precision, recall, F1, AUC)
  - Business metric: % predictions within ±10% of actual

## Reported Performance

**Note**: The following values appear in the notebook outputs but cannot be independently verified without access to the actual data:

- **Overall R²**: ~0.65 (65% variance explained)
- **MAPE**: ~71% (high percentage errors, especially for outliers)
- **Within ±10% accuracy**: ~83% of predictions
- **Classification accuracy**: 85-89% for identifying revenue-generating users

## Technical Concerns & Recommendations

### Strengths:
- Hybrid approach appropriately handles zero-inflation
- Comprehensive feature engineering pipeline
- Multiple model types tested (XGBoost, Neural Networks, Random Forest, CatBoost)
- Log transformation handles skewed revenue distributions

### Limitations:
- **Extreme extrapolation**: Predicting 6 months from 3 days of data is highly ambitious
- **No temporal validation**: Should use time-based splits for realistic evaluation
- **High MAPE values**: Indicates significant percentage errors in predictions
- **Data leakage risk**: Target encoding without proper cross-validation

### Recommendations for Improvement:
1. Implement time-based validation splits to test on future cohorts
2. Add confidence intervals for predictions given the extreme extrapolation
3. Include feature importance analysis to understand predictive drivers
4. Consider intermediate targets (D30, D60, D90) to validate extrapolation
5. Add business-oriented metrics like revenue capture rates

## Dependencies

Required packages (no requirements.txt provided):
```bash
pip install pandas numpy scikit-learn xgboost catboost lightgbm matplotlib seaborn boto3 category-encoders jupyter
```

## Model Files Generated

The notebooks save trained models as pickle files:
- `ltv_model_no_uninstalls_xgb.pkl` - XGBoost regression model
- `ltv_model_no_uninstalls_mlp.pkl` - MLP regression model
- `target_encoder.pkl` - Categorical encoding transformer
- `scaler.pkl` - Feature scaling transformer
- `pca.pkl` - Dimensionality reduction transformer
- `tfidf_{day_prefix}.pkl` - TF-IDF vectorizers

## Important Notes

1. **AWS Credentials Required**: Code expects AWS credentials for S3 access
2. **Large Memory Requirements**: Processing large datasets requires significant RAM
3. **Data Format**: Input data expected in specific CSV format with required columns
4. **Validation Caveat**: Displayed performance metrics are from previous notebook runs and cannot be verified without access to the actual dataset