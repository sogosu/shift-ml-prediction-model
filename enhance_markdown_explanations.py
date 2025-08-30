import json
import re

def get_detailed_explanation(source_lines, cell_index, all_cells):
    """Generate more detailed, context-aware explanations"""
    code = ''.join(source_lines).strip()
    
    if not code:
        return "### Empty Cell\n\nThis cell is currently empty. It may be used for testing code snippets or has been cleared after previous use."
    
    # More detailed analysis based on patterns
    explanations = []
    
    # === IMPORTS SECTION ===
    if 'import' in code or 'from' in code:
        explanations.append("### Library Import Section\n")
        
        if 'warnings' in code and 'filterwarnings' in code:
            explanations.append("**Warning Configuration**: This code configures how Python handles warnings during execution. By setting warnings to 'ignore', the notebook will run without displaying warning messages that might clutter the output. This is useful when you know certain warnings are not problematic for your analysis.")
        
        elif 'logging' in code:
            level = "INFO" if "INFO" in code else "WARNING" if "WARNING" in code else "DEBUG" if "DEBUG" in code else "default"
            explanations.append(f"**Logging Setup**: This establishes a logging system to track program execution. Logging at {level} level means the program will record important events and potential issues, helping developers understand what happens during code execution and troubleshoot problems if they arise.")
        
        elif 'pandas as pd' in code:
            explanations.append("**Pandas Library**: Importing pandas (nicknamed 'pd') provides powerful tools for data manipulation. Think of pandas as Excel on steroids - it can handle millions of rows, perform complex calculations, and transform data in ways that would be impossible or extremely slow in a spreadsheet application.")
        
        elif 'numpy as np' in code:
            explanations.append("**NumPy Library**: NumPy (nicknamed 'np') is the foundation for scientific computing in Python. It provides fast mathematical operations on large arrays of numbers, which is essential for machine learning algorithms that perform millions of calculations.")
        
        elif 'boto3' in code:
            explanations.append("**AWS Integration**: Boto3 is Amazon's official library for connecting to AWS services. This allows the notebook to download data from S3 (Amazon's cloud storage), upload results, and interact with other AWS services. The data for this analysis is stored in the cloud rather than locally.")
        
        elif 'sklearn' in code or 'scikit-learn' in code:
            tools = []
            if 'LabelEncoder' in code:
                tools.append("LabelEncoder (converts text categories to numbers)")
            if 'train_test_split' in code:
                tools.append("train_test_split (divides data for training and testing)")
            if 'StandardScaler' in code:
                tools.append("StandardScaler (normalizes numerical features)")
            if 'RandomForestRegressor' in code:
                tools.append("RandomForestRegressor (ensemble prediction model)")
            if 'classification_report' in code or 'roc_auc_score' in code:
                tools.append("evaluation metrics (measure model performance)")
            
            tools_str = ", ".join(tools) if tools else "machine learning tools"
            explanations.append(f"**Scikit-learn Components**: Importing {tools_str}. These tools form the backbone of the machine learning pipeline, handling everything from data preparation to model training and evaluation.")
        
        elif 'xgboost' in code:
            explanations.append("**XGBoost Algorithm**: XGBoost (Extreme Gradient Boosting) is one of the most successful machine learning algorithms for structured data. It builds multiple decision trees that work together, with each tree learning from the mistakes of previous trees. This algorithm often wins data science competitions due to its accuracy.")
        
        elif 'matplotlib' in code or 'pyplot' in code:
            explanations.append("**Matplotlib Visualization**: This library creates charts and graphs to visualize data and results. Visual representations help identify patterns, outliers, and relationships that might be missed when looking at raw numbers.")
        
        elif 'seaborn' in code:
            explanations.append("**Seaborn Visualization**: Seaborn builds on matplotlib to create more sophisticated statistical visualizations with less code. It automatically calculates statistical relationships and creates publication-quality graphs that reveal insights about the data.")
    
    # === DATA LOADING SECTION ===
    elif 's3://' in code or 'bucket' in code.lower() or 's3.download_file' in code:
        explanations.append("### Cloud Data Access\n")
        
        if 'user_data_key' in code or 'user_level_data' in code:
            explanations.append("**User Data Configuration**: This sets up access to user behavior data stored in Amazon S3. The data contains information about how users interact with the application over time, including their activities, engagement patterns, and potentially their purchasing behavior. The file path includes dates, suggesting this is regularly updated data.")
        
        elif 'revenue_data_key' in code or 'ad_ops_revenue' in code or 'query_revenue' in code:
            explanations.append("**Revenue Data Configuration**: This configures access to revenue data files. The system tracks multiple revenue streams (ad operations and query-based revenue), which need to be combined to get a complete picture of how much money each user generates. This financial data is crucial for predicting lifetime value.")
        
        elif 's3.download_file' in code:
            explanations.append("**Data Download**: This downloads data files from Amazon's cloud storage to the local machine for processing. The files are temporarily stored locally because it's faster to work with local files than to repeatedly access cloud storage. After analysis, results can be uploaded back to the cloud.")
    
    # === DATA PROCESSING SECTION ===
    elif 'pd.read_csv' in code:
        explanations.append("### Data Loading\n")
        explanations.append("**CSV File Reading**: This loads comma-separated value (CSV) data into a pandas DataFrame. A DataFrame is like a smart spreadsheet that can handle millions of rows and perform complex operations. The data is now in memory and ready for analysis. CSV is a common format for sharing data between different systems.")
    
    elif 'merge' in code or 'concat' in code or 'join' in code:
        explanations.append("### Data Combination\n")
        
        if 'merge' in code:
            merge_on = re.search(r"on=['\"](\w+)['\"]", code)
            key = merge_on.group(1) if merge_on else "a common key"
            explanations.append(f"**Data Merging**: This combines two datasets by matching rows based on {key}. It's like using VLOOKUP in Excel, but more powerful. For example, one dataset might have user IDs and behavior data, while another has user IDs and revenue data. Merging connects these to create a complete picture of each user.")
        
        elif 'concat' in code:
            explanations.append("**Data Concatenation**: This stacks multiple datasets together. Unlike merging (which matches rows), concatenation simply adds more rows or columns. This is useful when you have the same type of data from different time periods or sources that need to be combined into one large dataset.")
    
    # === FEATURE ENGINEERING SECTION ===
    elif 'feature' in code.lower() or 'transform' in code or 'encoder' in code.lower():
        explanations.append("### Feature Engineering\n")
        
        if 'StandardScaler' in code or 'scaler' in code.lower():
            explanations.append("**Feature Scaling**: This standardizes numerical features to have a mean of 0 and standard deviation of 1. Without scaling, features with large values (like revenue in dollars) would dominate features with small values (like click rates as decimals). Scaling ensures all features contribute equally to the model's decisions.")
        
        elif 'TargetEncoder' in code:
            explanations.append("**Categorical Encoding**: This converts text categories (like 'Premium User' or 'Free User') into numbers based on their relationship with what we're trying to predict. For example, if Premium Users typically generate $100 in revenue and Free Users generate $10, the encoder might assign 100 to Premium and 10 to Free. This is smarter than simple numbering because it preserves the predictive relationship.")
        
        elif 'TfidfVectorizer' in code or 'tfidf' in code.lower():
            explanations.append("**Text Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numbers by measuring word importance. Common words get low scores while unique, meaningful words get high scores. This helps the model understand text patterns, like whether certain keywords in user searches correlate with higher revenue.")
        
        elif 'PCA' in code:
            explanations.append("**Dimensionality Reduction**: Principal Component Analysis (PCA) combines many features into fewer, more meaningful ones. Imagine having 100 features about users - PCA might combine them into 20 super-features that capture 95% of the information. This makes models faster and often more accurate by focusing on the most important patterns.")
    
    # === MODEL TRAINING SECTION ===
    elif '.fit(' in code or 'train_test_split' in code:
        explanations.append("### Model Training\n")
        
        if 'train_test_split' in code:
            test_size = re.search(r"test_size=([\d.]+)", code)
            size = f"{float(test_size.group(1))*100:.0f}%" if test_size else "a portion"
            explanations.append(f"**Data Splitting**: This divides the data into training and testing sets. The model learns from the training data (like studying with practice problems) and is evaluated on the testing data (like taking a final exam with new questions). Using {size} for testing ensures the model works on new, unseen data, not just memorized answers.")
        
        elif 'RandomizedSearchCV' in code or 'GridSearchCV' in code:
            explanations.append("**Hyperparameter Optimization**: This automatically tests different model configurations to find the best settings. It's like trying different recipe variations (more sugar, less salt, different cooking time) to find the perfect combination. The computer tests hundreds of combinations and picks the one that performs best.")
        
        elif '.fit(' in code:
            if 'XGB' in code or 'xgb' in code:
                explanations.append("**XGBoost Model Training**: The model is now learning patterns from the training data. It builds a series of decision trees, where each tree learns from the errors of previous trees. This iterative improvement process continues until the model achieves optimal performance. The training might take several minutes depending on data size.")
            elif 'RandomForest' in code:
                explanations.append("**Random Forest Training**: This trains multiple decision trees simultaneously, each looking at different random subsets of the data. The trees then vote on predictions, like a panel of experts making a group decision. This approach is robust and rarely overfits (memorizes rather than learns).")
            elif 'MLP' in code or 'neural' in code.lower():
                explanations.append("**Neural Network Training**: This trains an artificial neural network, inspired by how human brains work. Data flows through layers of interconnected nodes, with each layer learning increasingly complex patterns. Neural networks can capture non-linear relationships that simpler models might miss.")
            else:
                explanations.append("**Model Training**: The machine learning model is now learning patterns from historical data. It analyzes relationships between features (inputs) and targets (outputs) to understand what factors influence the outcome. This learning process involves complex mathematical optimization to find the best parameters.")
    
    # === PREDICTION SECTION ===
    elif '.predict(' in code or '.predict_proba(' in code:
        explanations.append("### Making Predictions\n")
        
        if 'np.expm1' in code or 'np.log1p' in code:
            explanations.append("**Transformed Predictions**: The model makes predictions on log-transformed data (where large values are compressed), then converts them back to normal scale using expm1. This technique helps models handle data with extreme values, like revenue that ranges from $0 to $10,000+. The transformation makes patterns easier to learn.")
        elif '.predict_proba(' in code:
            explanations.append("**Probability Predictions**: Instead of just predicting yes/no, the model provides probability scores (like 75% chance of high revenue). These probabilities help make nuanced business decisions - you might treat users with 90% probability differently from those with 60% probability.")
        else:
            explanations.append("**Generating Predictions**: The trained model now applies its learned patterns to make predictions on new data. Each prediction is based on the patterns and relationships the model discovered during training. These predictions estimate future user behavior or revenue based on current characteristics.")
    
    # === EVALUATION SECTION ===
    elif 'score' in code.lower() or 'accuracy' in code.lower() or 'metric' in code.lower():
        explanations.append("### Model Evaluation\n")
        
        if 'r2_score' in code:
            explanations.append("**R-Squared Evaluation**: R-squared measures how well the model's predictions match actual values, ranging from 0 (poor) to 1 (perfect). An R² of 0.75 means the model explains 75% of the variation in the data. This metric helps determine if the model is useful for business decisions.")
        elif 'mean_absolute_percentage_error' in code or 'MAPE' in code:
            explanations.append("**Percentage Error Calculation**: MAPE (Mean Absolute Percentage Error) shows the average percentage difference between predictions and actual values. For example, 10% MAPE means predictions are typically off by 10%. This metric is intuitive for business users to understand model accuracy.")
        elif 'classification_report' in code:
            explanations.append("**Classification Performance Report**: This generates a detailed report showing precision (how many positive predictions were correct), recall (how many actual positives were found), and F1-score (balanced measure). These metrics help understand where the model succeeds and where it needs improvement.")
        elif 'roc_auc' in code:
            explanations.append("**ROC-AUC Score**: This measures the model's ability to distinguish between classes (like high-value vs low-value users). A score of 1.0 is perfect, 0.5 is random guessing. Higher scores mean the model reliably identifies which users will generate revenue.")
    
    # === VISUALIZATION SECTION ===
    elif 'plt.' in code or 'plot(' in code or 'sns.' in code:
        explanations.append("### Data Visualization\n")
        
        if 'scatter' in code:
            explanations.append("**Scatter Plot Creation**: This creates a scatter plot showing the relationship between two variables. Each point represents one data point. Patterns in the scatter plot reveal correlations - points forming an upward line indicate positive correlation, while scattered points suggest no relationship.")
        elif 'hist' in code:
            explanations.append("**Histogram Generation**: This shows the distribution of values for a variable. The height of each bar represents how many data points fall in that range. Histograms reveal whether data is normally distributed, skewed, or has unusual patterns that need investigation.")
        elif 'heatmap' in code:
            explanations.append("**Heatmap Visualization**: This creates a color-coded matrix showing relationships between variables. Darker colors typically indicate stronger relationships. Heatmaps are excellent for spotting patterns in correlation matrices or confusion matrices at a glance.")
        elif 'barplot' in code or 'bar(' in code:
            explanations.append("**Bar Chart Creation**: This compares values across different categories using bar heights. Bar charts are ideal for comparing metrics like average revenue per user segment or model performance across different groups.")
        else:
            explanations.append("**Creating Visualization**: This generates a graph or chart to visualize data patterns or model results. Visual representations make complex data easier to understand and help identify insights that might be missed in numerical tables.")
    
    # === SAVING/LOADING SECTION ===
    elif 'pickle' in code or '.pkl' in code or 'save' in code.lower():
        explanations.append("### Model Persistence\n")
        
        if 'dump' in code or 'wb' in code:
            explanations.append("**Saving Model/Data**: This saves the trained model or processed data to a file for future use. Saving prevents having to retrain the model every time you need predictions. The saved file contains all the learned patterns and can be loaded instantly to make predictions on new data.")
        elif 'load' in code or 'rb' in code:
            explanations.append("**Loading Model/Data**: This loads a previously saved model or data from disk. Loading is much faster than retraining, allowing you to use pre-trained models immediately. This is how models are deployed in production - train once, load and use many times.")
    
    # === DEFAULT DETAILED EXPLANATION ===
    else:
        explanations.append("### Data Processing Step\n")
        
        # Provide context based on variable names and operations
        if 'revenue' in code.lower():
            explanations.append("**Revenue Calculation**: This code processes financial data related to user revenue. Revenue is a key metric for predicting customer lifetime value. The calculations might include summing transactions, averaging daily revenue, or identifying high-value customer segments.")
        elif 'user' in code.lower() or 'customer' in code.lower():
            explanations.append("**User Data Processing**: This handles information about users or customers, such as their demographics, behavior patterns, or engagement metrics. Understanding user characteristics helps predict their future value and behavior.")
        elif 'date' in code.lower() or 'time' in code.lower() or 'day' in code.lower():
            explanations.append("**Temporal Data Processing**: This code works with time-based data, analyzing how metrics change over days, weeks, or months. Time patterns are crucial for understanding user lifecycle and predicting long-term value.")
        elif any(op in code for op in ['groupby', 'agg', 'sum', 'mean', 'count']):
            explanations.append("**Data Aggregation**: This summarizes data by calculating statistics like totals, averages, or counts. Aggregation reduces detailed records into meaningful summary metrics that models can use for prediction.")
        elif 'drop' in code or 'remove' in code:
            explanations.append("**Data Cleaning**: This removes unnecessary or problematic data. Cleaning might involve dropping duplicate records, removing outliers, or eliminating features that don't contribute to predictions. Clean data leads to more accurate models.")
        elif 'fillna' in code or 'isna' in code or 'null' in code.lower():
            explanations.append("**Missing Value Handling**: This deals with gaps in the data where values are missing. The code might fill missing values with averages, zeros, or estimates, or might remove records with too many missing values. Proper handling of missing data is crucial for model accuracy.")
        else:
            explanations.append("**Processing Logic**: This code performs calculations or transformations on the data. Each transformation prepares the data for machine learning by extracting meaningful patterns or formatting data appropriately for algorithms.")
    
    return '\n\n'.join(explanations)

def enhance_notebook_explanations(notebook_path):
    """Enhance the notebook with detailed explanations"""
    
    print(f"Enhancing {notebook_path}...")
    
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells = notebook['cells']
    enhanced_cells = []
    code_cell_count = 0
    
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown' and i > 0:
            # Check if previous cell was code
            prev_cell = cells[i-1] if i > 0 else None
            if prev_cell and prev_cell['cell_type'] == 'code':
                # Skip the auto-generated markdown, we'll replace it
                continue
        
        if cell['cell_type'] == 'code':
            code_cell_count += 1
            source = cell.get('source', [])
            
            # Generate detailed explanation
            explanation = get_detailed_explanation(source, i, cells)
            
            # Create enhanced markdown cell
            markdown_cell = {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [f"## Step {code_cell_count}: {explanation}"]
            }
            
            enhanced_cells.append(markdown_cell)
        
        # Add the original cell
        enhanced_cells.append(cell)
    
    # Update notebook
    notebook['cells'] = enhanced_cells
    
    # Save enhanced notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Enhanced {notebook_path} - Added detailed explanations for {code_cell_count} code cells")
    return code_cell_count

# Enhance both notebooks
notebook1 = 'ben_work/d1-d180-prediction-model_exclude_d1_uninstalls_v6_hybrid_refined.ipynb'
notebook2 = 'ben_work/d1-d180-prediction-model_v8_hybrid_revenue_events_only_d1_d3.ipynb'

enhance_notebook_explanations(notebook1)
enhance_notebook_explanations(notebook2)

print("\nEnhancement complete! The notebooks now have detailed explanations before each code cell.")