# Executive Summary: Day-180 LTV Notebook Review

This review covers two notebooks in `ben_work/` that attempt to predict Day-180 lifetime value (LTV) using a short observation window (~Days 1–3). Both implement a hybrid approach: a classifier to predict revenue occurrence followed by a regressor to estimate amount for revenue-positive users. REVIEW copies with per‑cell notes are saved alongside the originals with the `_REVIEW.ipynb` suffix.

## Strengths
- Clear hybrid framing for zero‑inflated targets (classification → regression).
- Solid tabular ML choices (XGBoost/CatBoost/LightGBM, MLP) and pragmatic feature engineering (TargetEncoder, TF‑IDF, scaling, PCA).
- Sensible artifact persistence and modular preprocessing blocks reused across steps.

## Key Risks and Gaps
- Temporal validity: Random `train_test_split` is not appropriate for D180 forecasting. Use time‑based splits (train on earlier cohorts, test on later).
- Leakage hazards: Target encoding and global transforms must be fit on training folds only (ideally via cross‑fitting). Verify no label/aggregate leakage across the split boundary.
- Data joins: Confirm join keys and cardinality; prevent row duplication and post‑merge leakage (e.g., joins on future-derived keys).
- Metrics: MAPE can be misleading with zeros/small values. Add sMAPE, MAE, pinball loss (quantiles), and “% within ±10% of actual D180 LTV”. Calibrate classification probabilities.
- Uncertainty: No confidence intervals/quantiles; decisions on users benefit from interval estimates, not point predictions alone.

## Model‑Specific Notes
- v6 (excludes Day‑1 uninstalls): May introduce selection bias if the target population at inference contains such users. If exclusion mirrors deployment filtering, document it and enforce consistently upstream.
- v8 (revenue‑events‑only D1–D3): Focused signals can help, but risk under‑representing non‑revenue behaviors that predict future value. Consider features that encode intent and early engagement breadth/depth.

## Recommendations (Priority Order)
1. Replace random splits with cohort/time‑based validation and rolling‑origin backtests; report stability across cohorts.
2. Cross‑fit all encoders/transformers; pipeline with `ColumnTransformer`/`Pipeline` and per‑fold fitting. Lock seeds for reproducibility.
3. Add calibrated classification (Platt/Isotonic) and joint decision rules (thresholds tuned via business utility). Report expected value lift.
4. Expand metrics: sMAPE/MAE/quantile loss; error stratification by cohorts and user segments; coverage of prediction intervals.
5. Version data and schema; document S3 paths and snapshot dates. Produce a lightweight `requirements.txt` and `environment.yml`.
6. Baselines: naive carry‑forward, mean/median by segment, and simple GLM/Poisson Tweedie for benchmarking.
7. Interpretability: SHAP/permutation importance; reason‑codes for business review. Monitor for drift post‑deployment.

Find the annotated notebooks here:
- `ben_work/d1-d180-prediction-model_exclude_d1_uninstalls_v6_hybrid_refined_REVIEW.ipynb`
- `ben_work/d1-d180-prediction-model_v8_hybrid_revenue_events_only_d1_d3_REVIEW.ipynb`
