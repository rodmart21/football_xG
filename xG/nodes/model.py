import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import warnings

class SimpleXGModel:
    """
    Simple Expected Goals (xG) model using logistic regression.
    Predicts probability of a shot resulting in a goal.
    """
    
    def __init__(self):
        # Added class_weight='balanced' to handle imbalanced data
        self.model = LogisticRegression(max_iter=1000, random_state=42)
                                        # class_weight='balanced')
        self.scaler = StandardScaler()  # Added scaler for feature normalization
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
            technique_dummies = pd.get_dummies(df['shot_technique'], prefix='technique', drop_first=True)
            X = pd.concat([X, technique_dummies], axis=1)
        
        # Shot body part (one-hot encoding)
        if 'shot_body_part' in df.columns:
            bodypart_dummies = pd.get_dummies(df['shot_body_part'], prefix='bodypart', drop_first=True)
            X = pd.concat([X, bodypart_dummies], axis=1)
        
        # Interaction features
        X['dist_angle_interaction'] = X['dist_to_goal'] * X['angle_to_goal']
        X['dist_squared'] = X['dist_to_goal'] ** 2
        
        # Create target variable if shot_outcome exists
        y = None
        if 'shot_outcome' in df.columns:
            y = (df['shot_outcome'] == 'Goal').astype(int)
        
        return X, y
    
    def calculate_brier_score(self, y_true, y_pred_proba):
        """
        Calculate Brier Score (mean squared error of probability predictions).
        Lower is better, ranges from 0 (perfect) to 1 (worst).
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        y_pred_proba : array-like
            Predicted probabilities
            
        Returns:
        --------
        float : Brier score
        """
        return np.mean((y_pred_proba - y_true) ** 2)
    
    def calculate_ece(self, y_true, y_pred_proba, n_bins=10):
        """
        Calculate Expected Calibration Error (ECE).
        Measures the difference between predicted probabilities and actual frequencies.
        Lower is better, ranges from 0 (perfectly calibrated) to 1.
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels (0 or 1)
        y_pred_proba : array-like
            Predicted probabilities
        n_bins : int
            Number of bins to divide predictions into
            
        Returns:
        --------
        float : Expected Calibration Error
        """
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                # Average predicted probability in this bin
                avg_confidence_in_bin = np.mean(y_pred_proba[in_bin])
                # Actual accuracy in this bin
                accuracy_in_bin = np.mean(y_true[in_bin])
                # Weighted absolute difference
                ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
        
        return ece
    
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
        
        # Fit scaler on training data and transform both train and test
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate all metrics
        brier_score = self.calculate_brier_score(y_test, y_pred_proba)
        ece = self.calculate_ece(y_test, y_pred_proba, n_bins=10)
        
        metrics = {
            'accuracy': self.model.score(X_test_scaled, y_test),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'brier_score': brier_score,
            'ece': ece,
            'goal_rate_actual': y_test.mean(),
            'goal_rate_predicted': y_pred_proba.mean()
        }
        
        print("=" * 50)
        print("MODEL TRAINING RESULTS")
        print("=" * 50)
        print(f"Accuracy:              {metrics['accuracy']:.4f}")
        print(f"ROC AUC:               {metrics['roc_auc']:.4f}")
        print(f"Log Loss:              {metrics['log_loss']:.4f}")
        print(f"Brier Score:           {metrics['brier_score']:.4f}")
        print(f"ECE (Calibration):     {metrics['ece']:.4f}")
        print(f"Actual Goal Rate:      {metrics['goal_rate_actual']:.4f}")
        print(f"Predicted Goal Rate:   {metrics['goal_rate_predicted']:.4f}")
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
        for idx, row in importance.head(20).iterrows():
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
        
        # Scale features before prediction
        X_scaled = self.scaler.transform(X)
        
        xg = self.model.predict_proba(X_scaled)[:, 1]
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


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')


class AdvancedXGModel:
    """
    Advanced Expected Goals (xG) model using XGBoost with Optuna hyperparameter tuning.
    Predicts probability of a shot resulting in a goal.
    """
    
    def __init__(self, n_trials=50, cv_folds=3):
        """
        Parameters:
        -----------
        n_trials : int
            Number of Optuna trials for hyperparameter search
        cv_folds : int
            Number of cross-validation folds during tuning
        """
        self.model = None
        self.is_trained = False
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = None
        self.study = None
        self.X_train_sample = None  # For SHAP values
        self.explainer = None  # SHAP explainer
        
    def prepare_features(self, df):
        """
        Prepare enhanced features for the model from shot data.
        
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
        
        # Basic geometric features
        X['dist_to_goal'] = df['dist_to_goal']
        X['angle_to_goal'] = df['angle_to_goal_rad']
        
        # Polynomial features for distance and angle
        X['dist_squared'] = X['dist_to_goal'] ** 2
        X['angle_squared'] = X['angle_to_goal'] ** 2
        
        # Interaction features
        X['dist_angle_interaction'] = X['dist_to_goal'] * X['angle_to_goal']
        X['dist_sq_angle'] = X['dist_squared'] * X['angle_to_goal']
        
        # Inverse features (closer = higher value)
        X['inverse_dist'] = 1 / (X['dist_to_goal'] + 0.1)
        X['inverse_dist_angle'] = X['inverse_dist'] * X['angle_to_goal']

        # Additional engineered features
        X['shot_difficulty'] = X['dist_to_goal'] * (1 + X['angle_to_goal'] / np.pi)
        # Goalkeeper angle (how much goal is "visible")
        X['goal_angle_visible'] = 2 * np.arctan(7.32 / (2 * X['dist_to_goal']))
        # Distance from center (shots from center are better)
        X['lateral_distance'] = abs(df['y'] - 40)  # Assuming pitch width ~80m

        # Shot technique (one-hot encoding)
        if 'shot_technique' in df.columns:
            technique_dummies = pd.get_dummies(df['shot_technique'], prefix='technique', drop_first=True)
            X = pd.concat([X, technique_dummies], axis=1)
        
        # Shot body part (one-hot encoding)
        if 'shot_body_part' in df.columns:
            bodypart_dummies = pd.get_dummies(df['shot_body_part'], prefix='bodypart', drop_first=True)
            X = pd.concat([X, bodypart_dummies], axis=1)
        
        # Create target variable if shot_outcome exists
        y = None
        if 'shot_outcome' in df.columns:
            y = (df['shot_outcome'] == 'Goal').astype(int)
        
        return X, y
    
    def _objective(self, trial, X_train, y_train):
        """
        Optuna objective function for hyperparameter optimization with cross-validation.
        """
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        model = xgb.XGBClassifier(**params)
        
        # Use cross-validation instead of single validation set
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.cv_folds, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return cv_scores.mean()
    
    def train(self, df, test_size=0.2, verbose=True):
        """
        Train the xG model with Optuna hyperparameter tuning.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data with shot_outcome column
        test_size : float
            Proportion of data to use for testing
        verbose : bool
            Whether to print training progress
            
        Returns:
        --------
        dict : Training metrics
        """
        # Prepare features
        X, y = self.prepare_features(df)
        
        if y is None:
            raise ValueError("DataFrame must contain 'shot_outcome' column for training")
        
        # Split data into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        if verbose:
            print("=" * 60)
            print("STARTING HYPERPARAMETER OPTIMIZATION")
            print("=" * 60)
            print(f"Total samples: {len(X)}")
            print(f"Train: {len(X_train)}, Test: {len(X_test)}")
            print(f"Goal rate: {y.mean():.2%}")
            print(f"Running {self.n_trials} Optuna trials with {self.cv_folds}-fold CV...")
            print("-" * 60)
        
        # Optuna hyperparameter search
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=self.n_trials,
            show_progress_bar=verbose
        )
        
        self.best_params = self.study.best_params
        
        if verbose:
            print("\n" + "=" * 60)
            print("BEST HYPERPARAMETERS FOUND")
            print("=" * 60)
            for param, value in self.best_params.items():
                print(f"{param:20s}: {value}")
            print("-" * 60)
        
        # Train final model with best parameters
        final_params = self.best_params.copy()
        final_params.update({
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        })
        
        self.model = xgb.XGBClassifier(**final_params)
        self.model.fit(X_train, y_train, verbose=False)
        self.is_trained = True
        
        # Store sample for SHAP (use up to 500 samples for efficiency)
        sample_size = min(500, len(X_train))
        self.X_train_sample = X_train.sample(n=sample_size, random_state=42)
        
        # Evaluate on test set
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': self.model.score(X_test, y_test),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'goal_rate_actual': y_test.mean(),
            'goal_rate_predicted': y_pred_proba.mean()
        }
        
        # Calculate ECE (Expected Calibration Error)
        metrics['ece'] = self._calculate_ece(y_test, y_pred_proba, n_bins=10)
        
        if verbose:
            print("\n" + "=" * 60)
            print("FINAL MODEL EVALUATION (TEST SET)")
            print("=" * 60)
            print(f"Accuracy:              {metrics['accuracy']:.4f}")
            print(f"ROC AUC:               {metrics['roc_auc']:.4f}")
            print(f"Log Loss:              {metrics['log_loss']:.4f}")
            print(f"Brier Score:           {metrics['brier_score']:.4f}")
            print(f"ECE (Calibration):     {metrics['ece']:.4f}")
            print(f"Actual Goal Rate:      {metrics['goal_rate_actual']:.4f}")
            print(f"Predicted Goal Rate:   {metrics['goal_rate_predicted']:.4f}")
            print("=" * 60)
            
            # Feature importance
            self._show_feature_importance()
        
        return metrics
    
    def _calculate_ece(self, y_true, y_pred_proba, n_bins=10):
        """
        Calculate Expected Calibration Error (ECE).
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_pred_proba : array-like
            Predicted probabilities
        n_bins : int
            Number of bins for calibration
            
        Returns:
        --------
        float : ECE score (lower is better, 0 is perfect calibration)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_pred_proba[mask].mean()
                bin_weight = mask.sum() / len(y_true)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def _show_feature_importance(self, top_n=10):
        """Display feature importance from XGBoost model."""
        if not self.is_trained:
            return
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        print(f"\nTOP {top_n} MOST IMPORTANT FEATURES:")
        print("-" * 60)
        for idx, row in importance.head(top_n).iterrows():
            print(f"{row['feature']:30s}: {row['importance']:.4f}")
        print("-" * 60)
    
    def _show_least_important_features(self, bottom_n=10):
        """Display least important features from XGBoost model."""
        if not self.is_trained:
            print("Model must be trained first.")
            return
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=True)
        
        print(f"\nBOTTOM {bottom_n} LEAST IMPORTANT FEATURES:")
        print("-" * 60)
        for idx, row in importance.head(bottom_n).iterrows():
            print(f"{row['feature']:30s}: {row['importance']:.4f}")
        print("-" * 60)

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

    def plot_xg_calibration_styled(self, df, bins=10):
        """
        Plot xG calibration curve matching the reference style.
        
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
            'is_goal': ['mean', 'count']
        }).reset_index()
        calibration.columns = ['xg_bin', 'xG_mean', 'goal_rate', 'count']
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
        ax.set_facecolor('white')
        
        # Plot perfect calibration line (dashed red)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration', zorder=1)
        
        # Plot actual calibration points (blue circles)
        scatter = ax.scatter(
            calibration['xG_mean'], 
            calibration['goal_rate'],
            s=calibration['count'] * 3,  # Size based on count
            alpha=0.7,
            color='steelblue',
            edgecolors='steelblue',
            linewidths=1.5,
            zorder=2
        )
        
        # Set labels and title
        ax.set_xlabel('Predicted xG', fontsize=13)
        ax.set_ylabel('Actual Goal Rate', fontsize=13)
        ax.set_title('xG Calibration Plot', fontsize=15, weight='bold', pad=15)
        
        # Set axis limits
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')
        ax.set_axisbelow(True)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        
        # Set tick parameters
        ax.tick_params(labelsize=11)
        
        # Make sure aspect ratio is equal for square appearance
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.show()
    
    def plot_shap_summary(self, max_display=15):
        """
        Plot SHAP summary showing feature importance and impact on predictions.
        
        Parameters:
        -----------
        max_display : int
            Maximum number of features to display
        """
        if not self.is_trained:
            print("Model must be trained first.")
            return
        
        print("Calculating SHAP values (this may take a moment)...")
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer.shap_values(self.X_train_sample)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_train_sample, 
                         max_display=max_display, show=False)
        plt.tight_layout()
        plt.show()
    
    def plot_shap_bar(self, max_display=15):
        """
        Plot SHAP bar chart showing mean absolute SHAP values (feature importance).
        
        Parameters:
        -----------
        max_display : int
            Maximum number of features to display
        """
        if not self.is_trained:
            print("Model must be trained first.")
            return
        
        if self.explainer is None:
            print("Calculating SHAP values (this may take a moment)...")
            self.explainer = shap.TreeExplainer(self.model)
        
        shap_values = self.explainer.shap_values(self.X_train_sample)
        
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_train_sample, 
                         plot_type="bar", max_display=max_display, show=False)
        plt.tight_layout()
        plt.show()
    
    def plot_reliability_curve(self, df, n_bins=10):
        """
        Plot reliability (calibration) curve with ECE visualization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Shot data with shot_outcome
        n_bins : int
            Number of bins for calibration
        """
        if 'shot_outcome' not in df.columns:
            print("Cannot plot reliability curve without shot_outcome column")
            return
        
        df_copy = self.add_xg_to_dataframe(df)
        df_copy['is_goal'] = (df_copy['shot_outcome'] == 'Goal').astype(int)
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        df_copy['xg_bin'] = pd.cut(df_copy['xG'], bins=bins)
        
        # Calculate actual vs predicted for each bin
        calibration = df_copy.groupby('xg_bin', observed=True).agg({
            'xG': ['mean', 'count'],
            'is_goal': 'mean'
        }).reset_index()
        calibration.columns = ['xg_bin', 'xG_mean', 'count', 'goal_rate']
        
        # Calculate ECE
        ece = self._calculate_ece(df_copy['is_goal'].values, df_copy['xG'].values, n_bins)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
        
        # Left plot: Reliability curve
        ax1.set_facecolor('white')
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration', zorder=1)
        
        # Plot reliability curve with error bars
        ax1.scatter(calibration['xG_mean'], calibration['goal_rate'],
                   s=calibration['count'] * 3, alpha=0.7, color='steelblue',
                   edgecolors='steelblue', linewidths=1.5, zorder=2)
        ax1.plot(calibration['xG_mean'], calibration['goal_rate'], 
                'steelblue', alpha=0.5, linewidth=2, zorder=1)
        
        ax1.set_xlabel('Predicted Probability (xG)', fontsize=12)
        ax1.set_ylabel('Actual Goal Rate', fontsize=12)
        ax1.set_title(f'Reliability Curve\nECE = {ece:.4f}', fontsize=13, weight='bold')
        ax1.set_xlim(-0.02, 1.02)
        ax1.set_ylim(-0.02, 1.02)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.set_aspect('equal', adjustable='box')
        
        # Right plot: Calibration error bars
        ax2.set_facecolor('white')
        calibration_error = calibration['goal_rate'] - calibration['xG_mean']
        colors = ['green' if x >= 0 else 'red' for x in calibration_error]
        
        ax2.barh(range(len(calibration)), calibration_error, color=colors, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.set_yticks(range(len(calibration)))
        ax2.set_yticklabels([f"{x:.2f}" for x in calibration['xG_mean']], fontsize=9)
        ax2.set_xlabel('Calibration Error (Actual - Predicted)', fontsize=12)
        ax2.set_ylabel('Predicted xG Bin', fontsize=12)
        ax2.set_title('Calibration Error by Bin', fontsize=13, weight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
        
        # Print calibration details
        print("\n" + "=" * 60)
        print("CALIBRATION ANALYSIS")
        print("=" * 60)
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
        print(f"Brier Score: {brier_score_loss(df_copy['is_goal'], df_copy['xG']):.4f}")
        print("\nPer-bin calibration:")
        print("-" * 60)
        for _, row in calibration.iterrows():
            error = row['goal_rate'] - row['xG_mean']
            print(f"Bin {row['xG_mean']:.2f}: Actual={row['goal_rate']:.3f}, "
                  f"Error={error:+.3f}, N={int(row['count'])}")
        print("=" * 60)
        
    def plot_optimization_history(self):
        """Plot the optimization history from Optuna study."""
        if self.study is None:
            print("No optimization study available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Optimization history
        trials_df = self.study.trials_dataframe()
        ax1.plot(trials_df['number'], trials_df['value'], marker='o', alpha=0.6)
        ax1.axhline(y=self.study.best_value, color='r', linestyle='--', 
                   label=f'Best: {self.study.best_value:.4f}')
        ax1.set_xlabel('Trial Number', fontsize=12)
        ax1.set_ylabel('ROC AUC Score', fontsize=12)
        ax1.set_title('Optimization History', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Parameter importance (if available)
        try:
            param_importance = optuna.importance.get_param_importances(self.study)
            params = list(param_importance.keys())[:8]  # Top 8
            values = [param_importance[p] for p in params]
            
            ax2.barh(params, values, alpha=0.7, color='steelblue')
            ax2.set_xlabel('Importance', fontsize=12)
            ax2.set_title('Hyperparameter Importance', fontsize=14, weight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        except:
            ax2.text(0.5, 0.5, 'Not enough trials\nfor importance analysis', 
                    ha='center', va='center', fontsize=12)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()