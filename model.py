import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class BelgianGPPredictor:
    """
    Race winner prediction model for Belgian Grand Prix.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        
    def prepare_training_data(self, historical_data, feature_engineer):
        """
        Prepare training data from historical results.
        
        Args:
            historical_data (dict): Historical race data
            feature_engineer (FeatureEngineer): Feature engineering instance
            
        Returns:
            tuple: (X, y) training data
        """
        if 'results' not in historical_data or historical_data['results'].empty:
            print("No historical results data available")
            return pd.DataFrame(), pd.Series()
        
        results = historical_data['results']
        
        # Create target variable (winner = 1, others = 0)
        if 'Position' in results.columns:
            y = (results['Position'] == 1).astype(int)
        else:
            # If no Position column, create dummy target
            y = pd.Series([0] * len(results), dtype=int)
        
        # Create features for each driver/year combination
        features_list = []
        
        for _, row in results.iterrows():
            driver = row.get('Driver', 'Unknown')
            year = row.get('Year', 2024)
            
            # Driver historical performance at Spa
            driver_spa_history = results[
                (results['Driver'] == driver) & (results['Year'] < year)
            ]
            
            # Current season performance (races before Spa)
            current_season = results[
                (results['Driver'] == driver) & (results['Year'] == year)
            ]
            
            # Team performance
            team_performance = results[
                (results['Team'] == row.get('Team', 'Unknown')) & (results['Year'] == year)
            ] if 'Team' in results.columns else pd.DataFrame()
            
            features = {
                'Driver': driver,
                'Year': year,
                'Team': row.get('Team', 'Unknown'),
                'GridPosition': row.get('GridPosition', 10),
                'Points': row.get('Points', 0),
                
                # Historical Spa performance
                'SpaRaces': len(driver_spa_history),
                'SpaWins': (driver_spa_history['Position'] == 1).sum() if not driver_spa_history.empty else 0,
                'SpaPodiums': (driver_spa_history['Position'] <= 3).sum() if not driver_spa_history.empty else 0,
                'SpaAvgPosition': driver_spa_history['Position'].mean() if not driver_spa_history.empty else 15,
                'SpaAvgPoints': driver_spa_history['Points'].mean() if not driver_spa_history.empty and 'Points' in driver_spa_history.columns else 0,
                
                # Current season form
                'SeasonRaces': len(current_season),
                'SeasonWins': (current_season['Position'] == 1).sum() if not current_season.empty else 0,
                'SeasonPodiums': (current_season['Position'] <= 3).sum() if not current_season.empty else 0,
                'SeasonAvgPosition': current_season['Position'].mean() if not current_season.empty else 15,
                'SeasonPoints': current_season['Points'].sum() if not current_season.empty and 'Points' in current_season.columns else 0,
                
                # Team performance
                'TeamSeasonWins': (team_performance['Position'] == 1).sum() if not team_performance.empty else 0,
                'TeamSeasonPodiums': (team_performance['Position'] <= 3).sum() if not team_performance.empty else 0,
                'TeamSeasonPoints': team_performance['Points'].sum() if not team_performance.empty and 'Points' in team_performance.columns else 0,
                
                # Qualifying position advantage
                'QualifyingAdvantage': max(0, 10 - row.get('GridPosition', 10)),
                
                # Championship position (estimated)
                'ChampionshipPosition': self._estimate_championship_position(driver, year, results),
                
                # Recent form (last 3 races)
                'RecentForm': self._calculate_recent_form(driver, year, results),
                
                # Track characteristics favor
                'TrackCharacteristicsFavor': self._calculate_track_favor(driver, driver_spa_history)
            }
            
            features_list.append(features)
        
        # Create DataFrame
        X = pd.DataFrame(features_list)
        
        # Align with target
        y = y.reset_index(drop=True)
        
        return X, y
    
    def _estimate_championship_position(self, driver, year, results):
        """Estimate championship position based on points."""
        year_results = results[results['Year'] == year]
        if 'Points' not in year_results.columns:
            return 10
        
        driver_points = year_results[year_results['Driver'] == driver]['Points'].sum()
        all_driver_points = year_results.groupby('Driver')['Points'].sum().sort_values(ascending=False)
        
        position = (all_driver_points > driver_points).sum() + 1
        return min(position, 20)
    
    def _calculate_recent_form(self, driver, year, results):
        """Calculate recent form score."""
        driver_results = results[
            (results['Driver'] == driver) & (results['Year'] == year)
        ].tail(3)
        
        if driver_results.empty:
            return 0
        
        # Weight recent races more heavily
        weights = [0.2, 0.3, 0.5]
        form_score = 0
        
        for i, (_, race) in enumerate(driver_results.iterrows()):
            position = race['Position']
            race_score = max(0, 21 - position)  # Points-like scoring
            form_score += race_score * weights[i] if i < len(weights) else race_score * 0.5
        
        return form_score
    
    def _calculate_track_favor(self, driver, spa_history):
        """Calculate how much the track characteristics favor this driver."""
        if spa_history.empty:
            return 0
        
        # Based on historical overperformance at Spa
        avg_position = spa_history['Position'].mean()
        # Assume average grid position is around 10 for comparison
        overperformance = max(0, 10 - avg_position)
        
        return overperformance
    
    def train_models(self, X, y):
        """
        Train multiple models and select the best one.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            dict: Training results
        """
        if X.empty or y.empty:
            print("No training data available")
            return {}
        
        # Prepare data
        X_processed = self._preprocess_features(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models
        models_config = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                verbosity=0
            )
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'accuracy': accuracy,
                'auc': auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Store model
            self.models[name] = model
            
            print(f"{name} - CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"{name} - Test Accuracy: {accuracy:.3f}, Test AUC: {auc:.3f}")
            print()
        
        # Select best model based on CV AUC
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"Best model: {best_model_name}")
        
        # Feature importance
        self._calculate_feature_importance(X_processed)
        
        return results
    
    def _preprocess_features(self, X):
        """
        Preprocess features for model training.
        
        Args:
            X (pd.DataFrame): Raw features
            
        Returns:
            pd.DataFrame: Processed features
        """
        X_processed = X.copy()
        
        # Handle categorical variables
        categorical_cols = ['Driver', 'Team']
        for col in categorical_cols:
            if col in X_processed.columns:
                # Create dummy variables
                dummies = pd.get_dummies(X_processed[col], prefix=col)
                X_processed = pd.concat([X_processed, dummies], axis=1)
                X_processed.drop(col, axis=1, inplace=True)
        
        # Handle missing values
        X_processed = X_processed.fillna(0)
        
        # Scale numerical features
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numerical_cols] = self.scaler.fit_transform(X_processed[numerical_cols])
        
        return X_processed
    
    def _calculate_feature_importance(self, X_processed):
        """Calculate and store feature importance."""
        if self.best_model is None:
            return
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # Linear models
            importance = np.abs(self.best_model.coef_[0])
        else:
            return
        
        self.feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def predict_winner_probabilities(self, prediction_features):
        """
        Predict winner probabilities for upcoming race.
        
        Args:
            prediction_features (pd.DataFrame): Features for prediction
            
        Returns:
            pd.DataFrame: Predictions with probabilities
        """
        if self.best_model is None:
            print("No model trained yet")
            return pd.DataFrame()
        
        # Preprocess features
        X_processed = self._preprocess_features(prediction_features)
        
        # Ensure all training columns are present
        try:
            training_columns = self.best_model.feature_names_in_ if hasattr(self.best_model, 'feature_names_in_') else X_processed.columns
        except:
            training_columns = X_processed.columns
        
        # Add missing columns with zeros
        for col in training_columns:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        # Reorder columns to match training
        X_processed = X_processed[training_columns]
        
        # Make predictions
        probabilities = self.best_model.predict_proba(X_processed)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Driver': prediction_features['Driver'],
            'WinProbability': probabilities,
            'Team': prediction_features['Team'] if 'Team' in prediction_features.columns else 'Unknown'
        })
        
        # Sort by probability
        results = results.sort_values('WinProbability', ascending=False)
        
        # Add rank
        results['Rank'] = range(1, len(results) + 1)
        
        return results
    
    def create_prediction_features(self, current_season_data, historical_data):
        """
        Create features for current drivers for prediction.
        
        Args:
            current_season_data (pd.DataFrame): Current season data
            historical_data (dict): Historical data
            
        Returns:
            pd.DataFrame: Prediction features
        """
        if current_season_data.empty:
            print("No current season data available")
            return pd.DataFrame()
        
        # Get unique drivers
        drivers = current_season_data['Driver'].unique()
        current_year = 2024
        
        features_list = []
        
        for driver in drivers:
            # Current season performance
            driver_current = current_season_data[current_season_data['Driver'] == driver]
            
            # Historical Spa performance
            spa_history = pd.DataFrame()
            if 'results' in historical_data and not historical_data['results'].empty:
                spa_history = historical_data['results'][
                    historical_data['results']['Driver'] == driver
                ]
            
            # Team info
            team = driver_current['Team'].iloc[0] if 'Team' in driver_current.columns and not driver_current.empty else 'Unknown'
            
            features = {
                'Driver': driver,
                'Year': current_year,
                'Team': team,
                'GridPosition': 10,  # Will be updated with actual qualifying results
                'Points': driver_current['Points'].sum() if 'Points' in driver_current.columns else 0,
                
                # Historical Spa performance
                'SpaRaces': len(spa_history),
                'SpaWins': (spa_history['Position'] == 1).sum() if not spa_history.empty else 0,
                'SpaPodiums': (spa_history['Position'] <= 3).sum() if not spa_history.empty else 0,
                'SpaAvgPosition': spa_history['Position'].mean() if not spa_history.empty else 15,
                'SpaAvgPoints': spa_history['Points'].mean() if not spa_history.empty and 'Points' in spa_history.columns else 0,
                
                # Current season form
                'SeasonRaces': len(driver_current),
                'SeasonWins': (driver_current['Position'] == 1).sum() if 'Position' in driver_current.columns else 0,
                'SeasonPodiums': (driver_current['Position'] <= 3).sum() if 'Position' in driver_current.columns else 0,
                'SeasonAvgPosition': driver_current['Position'].mean() if 'Position' in driver_current.columns else 15,
                'SeasonPoints': driver_current['Points'].sum() if 'Points' in driver_current.columns else 0,
                
                # Team performance
                'TeamSeasonWins': (current_season_data[current_season_data['Team'] == team]['Position'] == 1).sum() if 'Position' in current_season_data.columns else 0,
                'TeamSeasonPodiums': (current_season_data[current_season_data['Team'] == team]['Position'] <= 3).sum() if 'Position' in current_season_data.columns else 0,
                'TeamSeasonPoints': current_season_data[current_season_data['Team'] == team]['Points'].sum() if 'Points' in current_season_data.columns else 0,
                
                # Qualifying position advantage (will be updated)
                'QualifyingAdvantage': 0,
                
                # Championship position
                'ChampionshipPosition': self._estimate_championship_position_current(driver, current_season_data),
                
                # Recent form
                'RecentForm': self._calculate_recent_form_current(driver, current_season_data),
                
                # Track characteristics favor
                'TrackCharacteristicsFavor': self._calculate_track_favor(driver, spa_history)
            }
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _estimate_championship_position_current(self, driver, current_season_data):
        """Estimate current championship position."""
        if 'Points' not in current_season_data.columns:
            return 10
        
        driver_points = current_season_data[current_season_data['Driver'] == driver]['Points'].sum()
        all_driver_points = current_season_data.groupby('Driver')['Points'].sum().sort_values(ascending=False)
        
        position = (all_driver_points > driver_points).sum() + 1
        return min(position, 20)
    
    def _calculate_recent_form_current(self, driver, current_season_data):
        """Calculate recent form for current season."""
        driver_results = current_season_data[
            current_season_data['Driver'] == driver
        ].tail(3)
        
        if driver_results.empty or 'Position' not in driver_results.columns:
            return 0
        
        # Weight recent races more heavily
        weights = [0.2, 0.3, 0.5]
        form_score = 0
        
        for i, (_, race) in enumerate(driver_results.iterrows()):
            position = race['Position']
            race_score = max(0, 21 - position)  # Points-like scoring
            form_score += race_score * weights[i] if i < len(weights) else race_score * 0.5
        
        return form_score
    
    def get_model_summary(self):
        """
        Get summary of the trained model.
        
        Returns:
            dict: Model summary
        """
        if self.best_model is None:
            return {}
        
        summary = {
            'best_model': self.best_model_name,
            'feature_importance': self.feature_importance.head(10).to_dict('records') if self.feature_importance is not None else [],
            'model_params': self.best_model.get_params() if hasattr(self.best_model, 'get_params') else {}
        }
        
        return summary