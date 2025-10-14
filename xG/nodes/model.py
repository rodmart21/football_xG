import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, log_loss
import matplotlib.pyplot as plt

class SimpleXGModel:
    """
    Simple Expected Goals (xG) model using logistic regression.
    Predicts probability of a shot resulting in a goal.
    """
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False
    
    def prepare_features(self, df):
        """
        Prepare features for the model from shot data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with shot data
            
        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series (optional)
            Target variable (1 if goal, 0 otherwise) - only if shot_outcome exists
        """
        X = pd.DataFrame()
        
        # Distance to goal (already calculated)
        X['dist_to_goal'] = df['dist_to_goal']
        
        # Angle to goal (in radians)
        X['angle_to_goal'] = df['angle_to_goal_rad']
        
        # Shot technique (one-hot encoding)
        if 'shot_technique' in df.columns:
            technique_dummies = pd.get_dummies(df['shot_technique'], prefix='technique')
            X = pd.concat([X, technique_dummies], axis=1)
        
        # Shot body part (one-hot encoding)
        if 'shot_body_part' in df.columns:
            bodypart_dummies = pd.get_dummies(df['shot_body_part'], prefix='bodypart')
            X = pd.concat([X, bodypart_dummies], axis=1)
        
        # Interaction features
        X['dist_angle_interaction'] = X['dist_to_goal'] * X['angle_to_goal']
        X['dist_squared'] = X['dist_to_goal'] ** 2
        
        # Create target variable if shot_outcome exists
        y = None
        if 'shot_outcome' in df.columns:
            y = (df['shot_outcome'] == 'Goal').astype(int)
        
        return X, y
    
    def train(self, df, test_size=0.2):
        """
        Train the xG model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data with shot_outcome column
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        dict : Training metrics
        """
        # Prepare features
        X, y = self.prepare_features(df)
        
        if y is None:
            raise ValueError("DataFrame must contain 'shot_outcome' column for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': self.model.score(X_test, y_test),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'goal_rate_actual': y_test.mean(),
            'goal_rate_predicted': y_pred_proba.mean()
        }
        
        print("=" * 50)
        print("MODEL TRAINING RESULTS")
        print("=" * 50)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Log Loss: {metrics['log_loss']:.4f}")
        print(f"Actual Goal Rate: {metrics['goal_rate_actual']:.4f}")
        print(f"Predicted Goal Rate: {metrics['goal_rate_predicted']:.4f}")
        print("=" * 50)
        
        # Feature importance
        self._show_feature_importance()
        
        return metrics
    
    def _show_feature_importance(self):
        """Display feature importance from model coefficients."""
        if not self.is_trained:
            return
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0]
        })
        importance['abs_coefficient'] = importance['coefficient'].abs()
        importance = importance.sort_values('abs_coefficient', ascending=False)
        
        print("\nTOP 10 MOST IMPORTANT FEATURES:")
        print("-" * 50)
        for idx, row in importance.head(10).iterrows():
            print(f"{row['feature']:30s}: {row['coefficient']:+.4f}")
        print("-" * 50)
    
    def predict_xg(self, df):
        """
        Predict xG (probability of goal) for shots.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Shot data
            
        Returns:
        --------
        np.array : xG values (probability of goal)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        X, _ = self.prepare_features(df)
        
        # Ensure all training features are present
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Reorder columns to match training data
        X = X[self.feature_names]
        
        xg = self.model.predict_proba(X)[:, 1]
        return xg
    
    def add_xg_to_dataframe(self, df):
        """
        Add xG predictions to the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Shot data
            
        Returns:
        --------
        pd.DataFrame : Original dataframe with added 'xG' column
        """
        df_copy = df.copy()
        df_copy['xG'] = self.predict_xg(df)
        return df_copy
    
    def plot_xg_calibration(self, df, bins=10):
        """
        Plot xG calibration curve to see how well predictions match reality.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Shot data with shot_outcome
        bins : int
            Number of bins for calibration plot
        """
        if 'shot_outcome' not in df.columns:
            print("Cannot plot calibration without shot_outcome column")
            return
        
        df_copy = self.add_xg_to_dataframe(df)
        df_copy['is_goal'] = (df_copy['shot_outcome'] == 'Goal').astype(int)
        
        # Create bins
        df_copy['xg_bin'] = pd.cut(df_copy['xG'], bins=bins)
        
        # Calculate actual vs predicted for each bin
        calibration = df_copy.groupby('xg_bin', observed=True).agg({
            'xG': 'mean',
            'is_goal': 'mean'
        }).reset_index()
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(calibration['xG'], calibration['is_goal'], s=100, alpha=0.6)
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax.set_xlabel('Predicted xG', fontsize=12)
        ax.set_ylabel('Actual Goal Rate', fontsize=12)
        ax.set_title('xG Calibration Plot', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
