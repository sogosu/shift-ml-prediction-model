import json
import copy

def analyze_code_cell(source_lines):
    """Analyze code cell content and generate detailed explanation"""
    code = ''.join(source_lines).strip()
    
    # Skip empty cells
    if not code:
        return "This cell is empty and can be used for testing quick code snippets."
    
    explanation = []
    
    # Import statements
    if 'import' in code or 'from' in code:
        if 'warnings' in code:
            explanation.append("This cell sets up warning handling to control which warnings are displayed during code execution. Warnings are messages that alert you to potential issues that aren't errors but might need attention.")
        elif 'logging' in code:
            explanation.append("This cell configures logging, which is a way to track events that happen when the program runs. Logging helps developers understand what the program is doing and debug issues.")
        elif 'pandas' in code:
            explanation.append("This cell imports pandas, a powerful data manipulation library. Pandas is used to work with structured data in tables (similar to Excel spreadsheets).")
        elif 'numpy' in code:
            explanation.append("This cell imports NumPy, a library for numerical computing. NumPy provides tools for working with arrays and mathematical operations.")
        elif 'sklearn' in code or 'scikit-learn' in code:
            explanation.append("This cell imports components from scikit-learn, a machine learning library. These tools will be used to build and evaluate predictive models.")
        elif 'xgboost' in code or 'XGB' in code:
            explanation.append("This cell imports XGBoost, an advanced machine learning algorithm known for its high performance in prediction tasks.")
        elif 'catboost' in code:
            explanation.append("This cell imports CatBoost, a machine learning algorithm that handles categorical features well.")
        elif 'lightgbm' in code:
            explanation.append("This cell imports LightGBM, a fast and efficient gradient boosting framework for machine learning.")
        elif 'matplotlib' in code or 'pyplot' in code:
            explanation.append("This cell imports matplotlib, a library for creating visualizations and charts to help understand data patterns.")
        elif 'seaborn' in code:
            explanation.append("This cell imports Seaborn, a statistical data visualization library that makes attractive and informative graphs.")
        elif 'boto3' in code:
            explanation.append("This cell imports boto3, the Amazon Web Services (AWS) SDK for Python. This allows the code to interact with AWS services like S3 for data storage.")
        elif 'pickle' in code:
            explanation.append("This cell imports pickle, which is used to save and load Python objects (like trained models) to and from files.")
        else:
            explanation.append("This cell imports necessary libraries and modules that will be used throughout the notebook.")
    
    # Variable assignments and data loading
    elif 'pd.read_csv' in code:
        explanation.append("This cell reads data from a CSV (Comma-Separated Values) file into a pandas DataFrame. A DataFrame is like a spreadsheet or table where data is organized in rows and columns.")
    elif 's3.download_file' in code or 'S3' in code:
        explanation.append("This cell downloads data files from Amazon S3 (Simple Storage Service), which is a cloud storage service. The data is being retrieved from the cloud to be processed locally.")
    elif '=' in code and ('key' in code.lower() or 'path' in code.lower() or 'bucket' in code.lower()):
        explanation.append("This cell defines file paths, keys, or identifiers that specify where to find or store data. These are like addresses that tell the program where to look for files.")
    
    # Data processing
    elif 'df[' in code or 'DataFrame' in code or '.merge' in code or '.concat' in code:
        if 'merge' in code:
            explanation.append("This cell combines multiple datasets together based on common columns. This is like joining two Excel sheets based on a matching ID column.")
        elif 'concat' in code:
            explanation.append("This cell concatenates (stacks) multiple datasets together. This combines data by adding rows or columns from one dataset to another.")
        elif 'drop' in code:
            explanation.append("This cell removes unnecessary columns or rows from the dataset to keep only the relevant information for analysis.")
        elif 'fillna' in code or 'isna' in code or 'isnull' in code:
            explanation.append("This cell handles missing values in the data. Missing values are empty cells that need to be filled or removed for proper analysis.")
        elif 'groupby' in code:
            explanation.append("This cell groups data by certain categories and calculates summary statistics. This is like creating a pivot table in Excel.")
        elif 'pivot' in code:
            explanation.append("This cell reshapes the data by pivoting it, transforming rows into columns or vice versa for better analysis.")
        else:
            explanation.append("This cell processes and transforms the data, preparing it for analysis or machine learning.")
    
    # Feature engineering
    elif 'fit_transform' in code or 'transform' in code or 'fit(' in code:
        if 'StandardScaler' in code or 'scaler' in code.lower():
            explanation.append("This cell scales numerical features to have a standard range. This ensures all features contribute equally to the model, preventing features with large values from dominating.")
        elif 'TargetEncoder' in code or 'target_encoder' in code.lower():
            explanation.append("This cell encodes categorical variables (text categories) into numerical values based on their relationship with the target variable we're trying to predict.")
        elif 'TfidfVectorizer' in code or 'tfidf' in code.lower():
            explanation.append("This cell converts text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency). This technique helps identify important words in text.")
        elif 'PCA' in code or 'pca' in code.lower():
            explanation.append("This cell applies Principal Component Analysis (PCA) to reduce the number of features while preserving important information. This simplifies the data without losing key patterns.")
        else:
            explanation.append("This cell transforms features to prepare them for machine learning models.")
    
    # Model training
    elif '.fit(' in code and ('model' in code.lower() or 'clf' in code.lower() or 'reg' in code.lower()):
        explanation.append("This cell trains a machine learning model on the prepared data. Training means the model learns patterns from historical data to make future predictions.")
    elif 'train_test_split' in code:
        explanation.append("This cell splits the data into training and testing sets. The training set is used to teach the model, while the testing set is used to evaluate how well it performs on new, unseen data.")
    elif 'RandomizedSearchCV' in code or 'GridSearchCV' in code:
        explanation.append("This cell performs hyperparameter tuning to find the best settings for the machine learning model. This is like fine-tuning a recipe to get the best results.")
    
    # Model evaluation
    elif '.predict(' in code or '.predict_proba(' in code:
        explanation.append("This cell uses the trained model to make predictions on data. The model applies what it learned to generate predicted values.")
    elif 'accuracy' in code.lower() or 'r2_score' in code or 'mean_squared_error' in code or 'classification_report' in code:
        explanation.append("This cell evaluates the model's performance using metrics. These metrics tell us how accurate the model's predictions are compared to actual values.")
    elif 'confusion_matrix' in code:
        explanation.append("This cell creates a confusion matrix, which shows how many predictions were correct or incorrect for each category. This helps identify where the model makes mistakes.")
    
    # Visualization
    elif 'plt.' in code or 'plot' in code or 'scatter' in code or 'hist' in code:
        explanation.append("This cell creates visualizations to display data or results graphically. Charts and graphs make it easier to understand patterns and relationships in the data.")
    elif 'sns.' in code:
        explanation.append("This cell creates statistical visualizations using Seaborn. These plots help identify patterns, distributions, and relationships in the data.")
    
    # Model saving
    elif 'pickle.dump' in code or 'save(' in code or 'to_pickle' in code:
        explanation.append("This cell saves the trained model or processed data to a file. This allows the model to be reused later without retraining.")
    elif 'pickle.load' in code or 'load(' in code or 'read_pickle' in code:
        explanation.append("This cell loads a previously saved model or data from a file. This allows us to use pre-trained models without starting from scratch.")
    
    # Function definitions
    elif 'def ' in code:
        explanation.append("This cell defines a custom function that performs a specific task. Functions are reusable blocks of code that can be called multiple times with different inputs.")
    
    # Data exploration
    elif '.head()' in code or '.tail()' in code:
        explanation.append("This cell displays the first or last few rows of the dataset. This gives us a quick preview of what the data looks like.")
    elif '.info()' in code or '.describe()' in code:
        explanation.append("This cell shows summary information about the dataset, including data types, missing values, and statistical summaries. This helps understand the data structure and characteristics.")
    elif '.shape' in code:
        explanation.append("This cell checks the dimensions of the data (number of rows and columns). This tells us how much data we're working with.")
    elif '.value_counts()' in code:
        explanation.append("This cell counts the frequency of unique values in a column. This shows the distribution of categories or values in the data.")
    
    # Conditional logic
    elif 'if ' in code or 'for ' in code or 'while ' in code:
        if 'for ' in code:
            explanation.append("This cell contains a loop that repeats operations for multiple items. This automates repetitive tasks across the dataset.")
        elif 'if ' in code:
            explanation.append("This cell contains conditional logic that executes different code based on certain conditions. This allows the program to make decisions.")
        else:
            explanation.append("This cell contains control flow logic to manage how the code executes.")
    
    # Comments and documentation
    elif code.startswith('#') and len(code.splitlines()) <= 3:
        explanation.append("This cell contains comments or notes about the code. Comments help document what the code is doing and why certain decisions were made.")
    
    # Print statements
    elif 'print(' in code:
        explanation.append("This cell displays output to show results, progress, or debugging information. This helps track what the program is doing.")
    
    # Default explanation
    if not explanation:
        # Try to provide context based on variable names
        if 'revenue' in code.lower():
            explanation.append("This cell processes revenue-related data, which represents the money generated by users or transactions.")
        elif 'user' in code.lower():
            explanation.append("This cell processes user-related data, which contains information about the people using the service.")
        elif 'feature' in code.lower():
            explanation.append("This cell works with features (variables) that will be used to make predictions.")
        elif 'target' in code.lower() or 'label' in code.lower():
            explanation.append("This cell processes the target variable - the value we're trying to predict.")
        else:
            explanation.append("This cell performs data processing or analysis operations.")
    
    return ' '.join(explanation)

def add_markdown_explanations(notebook_path, output_path):
    """Add markdown explanations before each code cell"""
    
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Create new cells list with explanations
    new_cells = []
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Analyze the code and create explanation
            source = cell.get('source', [])
            explanation = analyze_code_cell(source)
            
            # Create markdown cell with explanation
            markdown_cell = {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [f"## What This Code Does\n\n{explanation}\n"]
            }
            
            # Add markdown cell before code cell
            new_cells.append(markdown_cell)
        
        # Add the original cell
        new_cells.append(cell)
    
    # Update notebook with new cells
    notebook['cells'] = new_cells
    
    # Save the updated notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Successfully added explanations to {output_path}")
    return len([c for c in new_cells if c['cell_type'] == 'markdown'])

# Process both notebooks
notebook1 = 'ben_work/d1-d180-prediction-model_exclude_d1_uninstalls_v6_hybrid_refined.ipynb'
notebook2 = 'ben_work/d1-d180-prediction-model_v8_hybrid_revenue_events_only_d1_d3.ipynb'

print("Processing first notebook...")
count1 = add_markdown_explanations(notebook1, notebook1)
print(f"Added {count1} markdown cells to first notebook")

print("\nProcessing second notebook...")
count2 = add_markdown_explanations(notebook2, notebook2)
print(f"Added {count2} markdown cells to second notebook")