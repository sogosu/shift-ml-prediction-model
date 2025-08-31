def srcstr: (.source // []) | join("");

def purpose($s):
  if ($s | test("warnings.*filterwarnings"; "i")) then "Configure warning handling to reduce noise"
  elif ($s | test("\\blogging\\b"; "i")) then "Configure logging for traceability"
  elif ($s | test("\\b(import|from)\\b"; "i")) then "Import libraries for data processing and modeling"
  elif ($s | test("s3://|boto3|s3\\.|download_file"; "i")) then "Access AWS S3 data sources"
  elif ($s | test("pd\\.read_|read_csv|read_parquet|read_json"; "i")) then "Load datasets into memory"
  elif ($s | test("merge|concat|join\\("; "i")) then "Combine datasets via joins/concatenation"
  elif ($s | test("fillna|dropna|isna|isnull"; "i")) then "Handle missing or null values"
  elif ($s | test("groupby|pivot|melt\\("; "i")) then "Aggregate or reshape data"
  elif ($s | test("StandardScaler|MinMaxScaler"; "i")) then "Scale/normalize numerical features"
  elif ($s | test("TargetEncoder|category_encoders"; "i")) then "Encode categorical features via target encoding"
  elif ($s | test("TfidfVectorizer|tfidf"; "i")) then "Vectorize text features using TF-IDF"
  elif ($s | test("PCA\\("; "i")) then "Reduce dimensionality with PCA"
  elif ($s | test("train_test_split"; "i")) then "Split data into train/test sets"
  elif ($s | test("XGBClassifier|XGBRegressor|xgboost"; "i")) then "Train XGBoost model(s)"
  elif ($s | test("CatBoost|LGBM"; "i")) then "Train gradient boosting model(s)"
  elif ($s | test("MLP|Keras|torch"; "i")) then "Train neural network model(s)"
  elif ($s | test("\\.fit\\("; "i")) then "Fit estimator/transformer on data"
  elif ($s | test("\\.predict|predict_proba"; "i")) then "Generate predictions/inferences"
  elif ($s | test("classification_report|roc_auc|r2_score|mean_squared_error|mape|rmse"; "i")) then "Evaluate model performance"
  elif ($s | test("pickle\\.dump|joblib\\.dump|to_pickle|save\\("; "i")) then "Persist models or artifacts to disk"
  elif ($s | test("plt\\.|seaborn|sns\\."; "i")) then "Visualize data or results"
  elif ($s | test("^\\s*$")) then "Empty cell / placeholder"
  elif ($s | test("^\\s*#")) then "Inline commentary / notes"
  else "General computation or housekeeping"
  end;

def reasoning($s):
  if ($s | test("TargetEncoder"; "i")) then "Reduce high-cardinality categorical noise by encoding signal relative to target"
  elif ($s | test("TfidfVectorizer|tfidf"; "i")) then "Convert sparse text into informative numeric features"
  elif ($s | test("PCA"; "i")) then "Condense correlated features and mitigate overfitting"
  elif ($s | test("StandardScaler|MinMaxScaler"; "i")) then "Put features on comparable scales for stable training"
  elif ($s | test("train_test_split"; "i")) then "Hold out data to estimate generalization"
  elif ($s | test("XGB|CatBoost|LGBM"; "i")) then "Use strong tabular learners for nonlinear interactions"
  elif ($s | test("MLP|Keras|torch"; "i")) then "Model complex relationships via neural nets"
  elif ($s | test("merge|join|concat"; "i")) then "Assemble a unified training table across sources"
  elif ($s | test("s3://|boto3"; "i")) then "Pull data from canonical cloud storage"
  elif ($s | test("pickle\\.dump|joblib\\.dump|save"; "i")) then "Enable reuse and deployment without retraining"
  else "Advance the hybrid pipeline toward D180 LTV prediction"
  end;

def validity($s):
  # Always include core cautions relevant to this project
  ( [
    (if ($s | test("TargetEncoder"; "i")) then "Guard against target leakage: fit encoders on training folds only" else empty end),
    (if ($s | test("train_test_split"; "i")) then "Prefer time-based splits for D180 forecasting over random splits" else empty end),
    (if ($s | test("TfidfVectorizer|tfidf|PCA|StandardScaler|MinMaxScaler|TargetEncoder"; "i")) then "Fit transforms on train only; apply to validation/test" else empty end),
    (if ($s | test("XGB|CatBoost|LGBM|MLP|Keras|torch"; "i")) then "Ensure reproducibility (fixed seeds) and calibrate classification probabilities" else empty end),
    (if ($s | test("r2_score|mape|rmse|classification_report|roc_auc"; "i")) then "Report business-relevant metrics: % within ±10% of actual D180 LTV" else empty end),
    (if ($s | test("merge|join"; "i")) then "Verify join keys, cardinalities, and prevent unintended row duplication" else empty end),
    (if ($s | test("s3://|boto3|download_file"; "i")) then "Pin data versions and schema; avoid mutating raw data locally" else empty end)
  ] + ["Method is reasonable given objectives; validate via cohort-based, time-ordered evaluation"]) | join("; ");

def review_cell($idx; $type; $s):
  {
    cell_type: "markdown",
    metadata: {review: true},
    source: [
      ("### Review: Cell " + ($idx|tostring) + " (" + $type + ")\n"),
      ("- Purpose: " + purpose($s) + "\n"),
      ("- Reasoning: " + reasoning($s) + "\n"),
      ("- Validity: " + validity($s) + "\n")
    ]
  };

def add_intro:
  {
    cell_type: "markdown",
    metadata: {review_copy: true},
    source: [
      "# REVIEW COPY\n",
      "This notebook is an annotated review. Each cell is preceded by reviewer notes: Purpose, Reasoning, and Validity for Day-180 LTV prediction.\n",
      "Notes are heuristic and based on static analysis; confirm with execution as needed.\n"
    ]
  };

def with_reviews:
  .cells as $cells
  | ($cells
      | to_entries
      | map([ review_cell(.key; .value.cell_type; (.value|srcstr)), .value ])
      | flatten
    ) as $newcells
  | .cells = ([ add_intro ] + $newcells);

with_reviews
