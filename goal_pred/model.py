import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def clean_data_and_remove_outliers(df, z_threshold=5, critical_cols=None):
    """
    Clean the dataset and remove outliers using a more lenient approach.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataset
    z_threshold : float, default=5
        Z-score threshold for outlier detection (higher = less strict)
    critical_cols : list, default=None
        List of critical columns to check for outliers. If None, only checks key stats.
    
    Returns:
    --------
    pandas DataFrame
        Cleaned dataset with outliers removed
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_size = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"Removed {initial_size - len(df_clean)} duplicate rows")
    
    # Fill missing values with median for numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Only check outliers on critical columns (more lenient approach)
    if critical_cols is None:
        # Only check key performance stats, not all columns
        critical_cols = ['Gls', 'Ast', 'xG', 'Min', '90s']
        critical_cols = [col for col in critical_cols if col in df_clean.columns]
    
    if len(critical_cols) > 0:
        # Calculate Z-scores only for critical columns
        z_scores = np.abs((df_clean[critical_cols] - df_clean[critical_cols].mean()) / df_clean[critical_cols].std())
        
        # Remove rows where ANY critical column has extreme outliers
        df_clean = df_clean[(z_scores < z_threshold).all(axis=1)]
    
    print(f"\nOriginal dataset size: {len(df)}")
    print(f"Cleaned dataset size: {len(df_clean)}")
    print(f"Removed {len(df) - len(df_clean)} rows ({((len(df) - len(df_clean))/len(df)*100):.2f}%)")
    
    return df_clean


def predict_goals(df, target_col='Gls', test_size=0.2, random_state=42):
    """
    Build a simple Random Forest model to predict goals.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The cleaned dataset
    target_col : str, default='Gls'
        The target column to predict (goals)
    test_size : float, default=0.2
        Proportion of dataset to use as test set
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing the trained model, predictions, and performance metrics
    """
    # Exclude target column and all its duplicates/derivatives
    # exclude_patterns = [
    #     'Rk', 'Player', 'Nation', 'Squad', 'Comp',  # Identification columns
    #     'Gls', 'G+A', 'G-PK', 'G-xG', 'G/Sh', 'G/SoT',  # Goal-related (target leakage)
    #     'np:G-xG', 'xG+xAG', 'G+A-PK',  # More goal derivatives
    #     'onG', 'Born'  # Other non-predictive columns
    # ]
    
    # # Get numeric columns only
    # numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Filter out excluded columns (including partial matches)
    feature_cols = ['Sh', 'SoT', '90s', 'Sh/90', 'SoT/90', 'SoT%', 'Att 3rd', 'PrgC']
    
    print(f"Using {len(feature_cols)} features for prediction")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    print("\n=== Model Performance ===")
    print(f"Train MAE: {train_mae:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print("\n=== Top 10 Most Important Features ===")
    print(feature_importance.to_string(index=False))
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'feature_importance': feature_importance,
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }