# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project for predicting customer lifetime value (LTV) and revenue over a 180-day period. The project uses hybrid models combining classification and regression approaches to predict user behavior and revenue generation.

## Project Structure

```
shift_ml_prediction_model/
└── ben_work/
    ├── d1-d180-prediction-model_exclude_d1_uninstalls_v6_hybrid_refined.ipynb
    └── d1-d180-prediction-model_v8_hybrid_revenue_events_only_d1_d3.ipynb
```

## Key Dependencies

The notebooks use the following ML and data processing libraries:
- **Data Processing**: pandas, numpy, boto3 (AWS S3 access)
- **Machine Learning**: scikit-learn, xgboost, catboost, lightgbm
- **Feature Engineering**: category_encoders (TargetEncoder), sklearn TfidfVectorizer, PCA
- **Visualization**: matplotlib, seaborn

## Development Commands

### Running Jupyter Notebooks
```bash
jupyter notebook
# or
jupyter lab
```

### Installing Dependencies
Since no requirements.txt exists, ensure these packages are installed:
```bash
pip install pandas numpy scikit-learn xgboost catboost lightgbm matplotlib seaborn boto3 category-encoders
```

## Architecture and Model Pipeline

### Data Pipeline
1. **Data Sources**: Pulls data from AWS S3 buckets
   - User-level data (Mixpanel events)
   - Ad operations revenue data
   - Query revenue data

2. **Feature Engineering**:
   - Text features using TF-IDF vectorization
   - Target encoding for categorical variables
   - StandardScaler for numerical features
   - PCA for dimensionality reduction

3. **Model Architecture** (Hybrid Approach):
   - **Classification Model**: XGBoost classifier to predict if users will generate revenue
   - **Regression Models**: 
     - XGBoost regressor for revenue prediction
     - MLP (Neural Network) regressor as alternative model
   - Models are trained on log-transformed revenue targets (using np.log1p)

### Key Model Files Generated
- `ltv_model_no_uninstalls_xgb.pkl` - XGBoost regression model
- `ltv_model_no_uninstalls_mlp.pkl` - MLP regression model  
- `target_encoder.pkl` - Categorical encoding transformer
- `scaler.pkl` - Feature scaling transformer
- `pca.pkl` - Dimensionality reduction transformer
- `tfidf_{day_prefix}.pkl` - TF-IDF vectorizers for text features

## Important Considerations

1. **AWS Credentials**: Code expects AWS credentials configured for S3 access
2. **Data Format**: Input data expected in CSV format with specific column structure
3. **Memory Usage**: Large datasets - consider memory constraints when running locally
4. **Model Versions**: Two notebook versions exist with different approaches:
   - v6: Excludes D1 uninstalls from training
   - v8: Focuses on revenue events only from days 1-3