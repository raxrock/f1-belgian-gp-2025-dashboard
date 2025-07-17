import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    """
    Feature engineering class for F1 Belgian GP data.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        
    def create_basic_features(self, laps_data):
        """
        Create basic features from lap data.
        
        Args:
            laps_data (pandas.DataFrame): Raw lap data
            
        Returns:
            pandas.DataFrame: Data with basic features
        """
        laps = laps_data.copy()
        
        # Convert lap time to seconds
        if 'LapTime' in laps.columns:
            laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        elif 'LapTimeSeconds' not in laps.columns:
            # If no lap time data, create a default column
            laps['LapTimeSeconds'] = 90.0  # Default lap time
        
        # Remove invalid laps if LapTimeSeconds exists
        if 'LapTimeSeconds' in laps.columns:
            laps = laps[laps['LapTimeSeconds'].notna()]
            laps = laps[laps['LapTimeSeconds'] > 60]  # Remove unrealistic lap times
        
        # Calculate stint length - handle missing columns gracefully
        required_cols = ['Driver', 'Year', 'Stint', 'LapNumber']
        missing_cols = [col for col in required_cols if col not in laps.columns]
        
        if not missing_cols:
            laps['StintLength'] = laps.groupby(['Driver', 'Year', 'Stint'])['LapNumber'].transform('count')
        else:
            laps['StintLength'] = 10  # Default stint length
        
        # Calculate tire degradation (lap time increase per lap in stint)
        if 'LapTimeSeconds' in laps.columns and not missing_cols:
            laps['TireDegradation'] = laps.groupby(['Driver', 'Year', 'Stint'])['LapTimeSeconds'].transform(
                lambda x: x.diff().rolling(window=3, min_periods=1).mean()
            )
        else:
            laps['TireDegradation'] = 0.1  # Default tire degradation
        
        # Calculate pace relative to stint start
        laps['PaceRelativeToStintStart'] = laps.groupby(['Driver', 'Year', 'Stint'])['LapTimeSeconds'].transform(
            lambda x: x - x.iloc[0] if len(x) > 0 else 0
        )
        
        # Calculate position changes
        laps['PositionChange'] = laps.groupby(['Driver', 'Year'])['Position'].transform('diff')
        
        # Track status indicators
        laps['SafetyCarPeriod'] = laps['TrackStatus'].isin(['4', '5', '6', '7'])
        
        # Tire age when stint started
        laps['TireAgeAtStintStart'] = laps.groupby(['Driver', 'Year', 'Stint'])['TireAge'].transform('first')
        
        return laps
    
    def create_driver_features(self, laps_data, results_data):
        """
        Create driver-specific features.
        
        Args:
            laps_data (pandas.DataFrame): Lap data
            results_data (pandas.DataFrame): Results data
            
        Returns:
            pandas.DataFrame: Driver features
        """
        # Ensure required columns exist
        if 'StintLength' not in laps_data.columns:
            laps_data['StintLength'] = 10  # Default stint length
        if 'TireDegradation' not in laps_data.columns:
            laps_data['TireDegradation'] = 0.1  # Default tire degradation
        
        # Driver performance metrics
        agg_dict = {}
        if 'LapTimeSeconds' in laps_data.columns:
            agg_dict['LapTimeSeconds'] = ['mean', 'std', 'min']
        if 'Position' in laps_data.columns:
            agg_dict['Position'] = ['mean', 'std']
        if 'TireDegradation' in laps_data.columns:
            agg_dict['TireDegradation'] = ['mean', 'std']
        if 'StintLength' in laps_data.columns:
            agg_dict['StintLength'] = ['mean', 'max']
        if 'Stint' in laps_data.columns:
            agg_dict['Stint'] = 'max'
        
        if not agg_dict:
            # If no aggregatable columns, create a basic dataframe
            unique_drivers = laps_data[['Driver', 'Year']].drop_duplicates() if 'Driver' in laps_data.columns and 'Year' in laps_data.columns else pd.DataFrame()
            return unique_drivers
        
        driver_stats = laps_data.groupby(['Driver', 'Year']).agg(agg_dict).reset_index()
        
        # Flatten column names
        driver_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in driver_stats.columns]
        driver_stats = driver_stats.rename(columns={'Driver_': 'Driver', 'Year_': 'Year'})
        
        # Add qualifying performance
        if 'GridPosition' in results_data.columns:
            quali_stats = results_data.groupby(['Driver', 'Year'])['GridPosition'].first().reset_index()
            driver_stats = driver_stats.merge(quali_stats, on=['Driver', 'Year'], how='left')
        
        # Add race results
        if 'Position' in results_data.columns:
            race_results = results_data.groupby(['Driver', 'Year'])['Position'].first().reset_index()
            race_results = race_results.rename(columns={'Position': 'FinalPosition'})
            driver_stats = driver_stats.merge(race_results, on=['Driver', 'Year'], how='left')
        
        # Calculate historical Spa performance
        spa_performance = results_data.groupby('Driver').agg({
            'Position': ['mean', 'std', 'count'],
            'Points': ['mean', 'sum'] if 'Points' in results_data.columns else ['mean', 'sum']
        }).reset_index()
        
        spa_performance.columns = ['Driver'] + [f'Spa_{col[0]}_{col[1]}' for col in spa_performance.columns[1:]]
        driver_stats = driver_stats.merge(spa_performance, on='Driver', how='left')
        
        return driver_stats
    
    def create_team_features(self, laps_data, results_data):
        """
        Create team-specific features.
        
        Args:
            laps_data (pandas.DataFrame): Lap data
            results_data (pandas.DataFrame): Results data
            
        Returns:
            pandas.DataFrame: Team features
        """
        # Ensure required columns exist
        if 'StintLength' not in laps_data.columns:
            laps_data['StintLength'] = 10  # Default stint length
        if 'TireDegradation' not in laps_data.columns:
            laps_data['TireDegradation'] = 0.1  # Default tire degradation
        
        # Check if Team column exists
        if 'Team' not in laps_data.columns:
            return pd.DataFrame()
        
        # Team performance metrics
        agg_dict = {}
        if 'LapTimeSeconds' in laps_data.columns:
            agg_dict['LapTimeSeconds'] = ['mean', 'std', 'min']
        if 'Position' in laps_data.columns:
            agg_dict['Position'] = ['mean', 'std']
        if 'TireDegradation' in laps_data.columns:
            agg_dict['TireDegradation'] = ['mean', 'std']
        if 'StintLength' in laps_data.columns:
            agg_dict['StintLength'] = ['mean', 'max']
        
        if not agg_dict:
            # If no aggregatable columns, create a basic dataframe
            unique_teams = laps_data[['Team', 'Year']].drop_duplicates() if 'Team' in laps_data.columns and 'Year' in laps_data.columns else pd.DataFrame()
            return unique_teams
        
        team_stats = laps_data.groupby(['Team', 'Year']).agg(agg_dict).reset_index()
        
        # Flatten column names
        team_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in team_stats.columns]
        team_stats = team_stats.rename(columns={'Team_': 'Team', 'Year_': 'Year'})
        
        # Add constructor standings points
        if 'Points' in results_data.columns:
            constructor_points = results_data.groupby(['Team', 'Year'])['Points'].sum().reset_index()
            constructor_points = constructor_points.rename(columns={'Points': 'TeamPoints'})
            team_stats = team_stats.merge(constructor_points, on=['Team', 'Year'], how='left')
        
        return team_stats
    
    def create_tire_features(self, laps_data, pit_stops_data):
        """
        Create tire-specific features.
        
        Args:
            laps_data (pandas.DataFrame): Lap data
            pit_stops_data (pandas.DataFrame): Pit stop data
            
        Returns:
            pandas.DataFrame: Tire features
        """
        # Tire compound performance
        tire_performance = laps_data.groupby(['Compound', 'Year']).agg({
            'LapTimeSeconds': ['mean', 'std', 'min'],
            'TireDegradation': ['mean', 'std'],
            'StintLength': ['mean', 'max'],
            'TireAge': ['mean', 'max']
        }).reset_index()
        
        # Flatten column names
        tire_performance.columns = ['_'.join(col).strip() if col[1] else col[0] for col in tire_performance.columns]
        tire_performance = tire_performance.rename(columns={'Compound_': 'Compound', 'Year_': 'Year'})
        
        # Pit stop strategy analysis
        if not pit_stops_data.empty:
            pit_strategy = pit_stops_data.groupby(['Driver', 'Year']).agg({
                'Lap': ['count', 'mean', 'std'],
                'PitTime': ['mean', 'std'] if 'PitTime' in pit_stops_data.columns else ['mean', 'std']
            }).reset_index()
            
            pit_strategy.columns = ['_'.join(col).strip() if col[1] else col[0] for col in pit_strategy.columns]
            pit_strategy = pit_strategy.rename(columns={'Driver_': 'Driver', 'Year_': 'Year'})
            
            # Calculate optimal pit windows
            pit_windows = pit_stops_data.groupby(['Year', 'Stint']).agg({
                'Lap': ['mean', 'std']
            }).reset_index()
            
            pit_windows.columns = ['Year', 'Stint'] + [f'OptimalPitWindow_{col[1]}' for col in pit_windows.columns[2:]]
            
            return tire_performance, pit_strategy, pit_windows
        
        return tire_performance, pd.DataFrame(), pd.DataFrame()
    
    def create_weather_features(self, laps_data):
        """
        Create weather-related features.
        
        Args:
            laps_data (pandas.DataFrame): Lap data with weather info
            
        Returns:
            pandas.DataFrame: Weather features
        """
        weather_features = laps_data.copy()
        
        # Weather impact on performance
        if 'Rainfall' in weather_features.columns:
            weather_features['IsWet'] = weather_features['Rainfall'] > 0
            weather_features['WeatherImpact'] = weather_features.groupby(['Driver', 'Year'])['LapTimeSeconds'].transform(
                lambda x: x.rolling(window=5, min_periods=1).std()
            )
        
        # Track temperature impact
        if 'TrackTemp' in weather_features.columns:
            weather_features['TempImpact'] = weather_features['TrackTemp'] - weather_features['TrackTemp'].mean()
        
        return weather_features
    
    def create_strategic_features(self, laps_data, pit_stops_data):
        """
        Create strategic features for undercut/overcut analysis.
        
        Args:
            laps_data (pandas.DataFrame): Lap data
            pit_stops_data (pandas.DataFrame): Pit stop data
            
        Returns:
            pandas.DataFrame: Strategic features
        """
        strategic_features = laps_data.copy()
        
        # Undercut/overcut opportunities
        if not pit_stops_data.empty:
            # Calculate gap to car ahead before pit stop
            strategic_features['GapToCarAhead'] = strategic_features.groupby(['Year', 'LapNumber'])['Position'].transform(
                lambda x: x.shift(-1) - x
            )
            
            # Calculate lap time advantage after pit stop
            strategic_features['PostPitAdvantage'] = strategic_features.groupby(['Driver', 'Year', 'Stint'])['LapTimeSeconds'].transform(
                lambda x: x.iloc[0] - x.shift(1).iloc[0] if len(x) > 1 else 0
            )
            
            # Strategic position (in DRS zone, etc.)
            strategic_features['InDRSZone'] = strategic_features['GapToCarAhead'] <= 1.0
        
        return strategic_features
    
    def create_prediction_features(self, current_season_data, historical_data):
        """
        Create features for race winner prediction.
        
        Args:
            current_season_data (pandas.DataFrame): Current season data
            historical_data (dict): Historical data dictionary
            
        Returns:
            pandas.DataFrame: Prediction features
        """
        prediction_features = []
        
        # Get unique drivers from current season
        if current_season_data.empty:
            return pd.DataFrame()
        
        drivers = current_season_data['Driver'].unique()
        
        for driver in drivers:
            features = {}
            features['Driver'] = driver
            
            # Current season performance
            driver_current = current_season_data[current_season_data['Driver'] == driver]
            
            if not driver_current.empty:
                features['CurrentSeasonPoints'] = driver_current['Points'].sum() if 'Points' in driver_current.columns else 0
                features['CurrentSeasonAvgPosition'] = driver_current['Position'].mean() if 'Position' in driver_current.columns else 20
                features['CurrentSeasonWins'] = (driver_current['Position'] == 1).sum() if 'Position' in driver_current.columns else 0
                features['CurrentSeasonPodiums'] = (driver_current['Position'] <= 3).sum() if 'Position' in driver_current.columns else 0
                
                # Recent form (last 5 races)
                recent_races = driver_current.tail(5)
                features['RecentAvgPosition'] = recent_races['Position'].mean() if 'Position' in recent_races.columns else 20
                features['RecentPoints'] = recent_races['Points'].sum() if 'Points' in recent_races.columns else 0
            
            # Historical Spa performance
            if 'results' in historical_data:
                spa_history = historical_data['results'][historical_data['results']['Driver'] == driver]
                if not spa_history.empty:
                    features['SpaHistoricalAvgPosition'] = spa_history['Position'].mean() if 'Position' in spa_history.columns else 20
                    features['SpaHistoricalWins'] = (spa_history['Position'] == 1).sum() if 'Position' in spa_history.columns else 0
                    features['SpaHistoricalPodiums'] = (spa_history['Position'] <= 3).sum() if 'Position' in spa_history.columns else 0
                    features['SpaRaceCount'] = len(spa_history)
                else:
                    features['SpaHistoricalAvgPosition'] = 20
                    features['SpaHistoricalWins'] = 0
                    features['SpaHistoricalPodiums'] = 0
                    features['SpaRaceCount'] = 0
            
            # Team performance
            if not driver_current.empty:
                team = driver_current['Team'].iloc[0] if 'Team' in driver_current.columns else 'Unknown'
                features['Team'] = team
                
                team_current = current_season_data[current_season_data['Team'] == team]
                features['TeamCurrentSeasonPoints'] = team_current['Points'].sum() if 'Points' in team_current.columns else 0
                features['TeamCurrentSeasonAvgPosition'] = team_current['Position'].mean() if 'Position' in team_current.columns else 20
            
            prediction_features.append(features)
        
        return pd.DataFrame(prediction_features)
    
    def preprocess_features(self, features_df):
        """
        Preprocess features for machine learning.
        
        Args:
            features_df (pandas.DataFrame): Features dataframe
            
        Returns:
            pandas.DataFrame: Preprocessed features
        """
        df = features_df.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        # Encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['Driver']:  # Keep driver names for identification
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Scale numerical features
        feature_columns = [col for col in numeric_columns if col not in ['Driver']]
        if feature_columns:
            df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        return df
    
    def create_target_variable(self, results_data):
        """
        Create target variable for prediction (race winner).
        
        Args:
            results_data (pandas.DataFrame): Results data
            
        Returns:
            pandas.DataFrame: Target variable
        """
        target = results_data.copy()
        target['Winner'] = (target['Position'] == 1).astype(int)
        return target[['Driver', 'Year', 'Winner']]
