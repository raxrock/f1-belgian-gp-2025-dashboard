import fastf1 as ff1
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class BelgianGPDataLoader:
    """
    Data loader for Belgian Grand Prix historical data using FastF1 API.
    """
    
    def __init__(self, years=None):
        """
        Initialize the data loader.
        
        Args:
            years (list): List of years to fetch data for. Default is 2018-2024.
        """
        self.years = years or list(range(2018, 2025))
        ff1.Cache.enable_cache('/tmp/fastf1_cache')
        
    def fetch_race_data(self, year):
        """
        Fetch race data for a specific year.
        
        Args:
            year (int): Year to fetch data for
            
        Returns:
            dict: Dictionary containing race data
        """
        try:
            session = ff1.get_session(year, 'Belgium', 'R')
            session.load()
            
            # Get lap data
            laps = session.laps
            
            # Get results
            results = session.results
            
            # Get weather data
            weather = session.weather_data
            
            # Get telemetry for tire data
            drivers = session.drivers
            
            return {
                'year': year,
                'laps': laps,
                'results': results,
                'weather': weather,
                'drivers': drivers,
                'session': session
            }
        except Exception as e:
            print(f"Error fetching data for {year}: {e}")
            return None
    
    def fetch_qualifying_data(self, year):
        """
        Fetch qualifying data for a specific year.
        
        Args:
            year (int): Year to fetch data for
            
        Returns:
            pandas.DataFrame: Qualifying results
        """
        try:
            session = ff1.get_session(year, 'Belgium', 'Q')
            session.load()
            return session.results
        except Exception as e:
            print(f"Error fetching qualifying data for {year}: {e}")
            return None
    
    def process_lap_data(self, race_data):
        """
        Process lap data to extract relevant features.
        
        Args:
            race_data (dict): Race data from fetch_race_data
            
        Returns:
            pandas.DataFrame: Processed lap data
        """
        if not race_data:
            return pd.DataFrame()
            
        laps = race_data['laps'].copy()
        year = race_data['year']
        
        # Add year column
        laps['Year'] = year
        
        # Calculate tire age (using stint-based calculation since TireLife column doesn't exist)
        laps['TireAge'] = laps.groupby(['Driver', 'Stint'])['LapNumber'].transform(lambda x: x - x.min() + 1)
        
        # Calculate lap time in seconds
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        
        # Add stint information
        laps['Stint'] = laps['Stint']
        
        # Calculate lap time delta to leader
        fastest_lap = laps['LapTimeSeconds'].min()
        laps['LapTimeDelta'] = laps['LapTimeSeconds'] - fastest_lap
        
        # Add tire compound information
        laps['Compound'] = laps['Compound']
        
        # Add position information
        laps['Position'] = laps['Position']
        
        # Add track status (for safety car detection)
        laps['TrackStatus'] = laps['TrackStatus']
        
        return laps
    
    def extract_pit_stops(self, race_data):
        """
        Extract pit stop information from race data.
        
        Args:
            race_data (dict): Race data from fetch_race_data
            
        Returns:
            pandas.DataFrame: Pit stop data
        """
        if not race_data:
            return pd.DataFrame()
            
        try:
            laps = race_data['laps']
            drivers = race_data['drivers']
            
            pit_stops = []
            
            for driver in drivers:
                driver_laps = laps[laps['Driver'] == driver]
                
                # Find pit stops (where stint changes)
                stint_changes = driver_laps[driver_laps['Stint'].diff() != 0]
                
                for idx, lap in stint_changes.iterrows():
                    if lap['Stint'] > 1:  # Skip first stint
                        pit_stops.append({
                            'Driver': driver,
                            'Year': race_data['year'],
                            'Lap': lap['LapNumber'],
                            'Stint': lap['Stint'],
                            'InLap': lap['LapNumber'] - 1,
                            'OutLap': lap['LapNumber'],
                            'TireCompound': lap['Compound'],
                            'PitTime': lap['PitOutTime'] - lap['PitInTime'] if pd.notna(lap['PitInTime']) else None
                        })
            
            return pd.DataFrame(pit_stops)
            
        except Exception as e:
            print(f"Error extracting pit stops: {e}")
            return pd.DataFrame()
    
    def merge_all_data(self):
        """
        Fetch and merge all historical data for Belgian GP.
        
        Returns:
            dict: Dictionary containing merged dataframes
        """
        all_lap_data = []
        all_qualifying_data = []
        all_pit_stops = []
        all_results = []
        
        print("Fetching historical Belgian GP data...")
        
        for year in tqdm(self.years, desc="Processing years"):
            # Fetch race data
            race_data = self.fetch_race_data(year)
            if race_data:
                # Process lap data
                lap_data = self.process_lap_data(race_data)
                if not lap_data.empty:
                    all_lap_data.append(lap_data)
                
                # Extract pit stops
                pit_stops = self.extract_pit_stops(race_data)
                if not pit_stops.empty:
                    all_pit_stops.append(pit_stops)
                
                # Add results
                results = race_data['results'].copy()
                results['Year'] = year
                
                # Map driver information - add Driver column from Abbreviation
                results['Driver'] = results['Abbreviation']
                results['Team'] = results['TeamName']
                
                all_results.append(results)
            
            # Fetch qualifying data
            quali_data = self.fetch_qualifying_data(year)
            if quali_data is not None:
                quali_data['Year'] = year
                
                # Map driver information - add Driver column from Abbreviation
                quali_data['Driver'] = quali_data['Abbreviation']
                quali_data['Team'] = quali_data['TeamName']
                
                all_qualifying_data.append(quali_data)
        
        # Combine all data
        merged_data = {}
        
        if all_lap_data:
            merged_data['laps'] = pd.concat(all_lap_data, ignore_index=True)
            
        if all_qualifying_data:
            merged_data['qualifying'] = pd.concat(all_qualifying_data, ignore_index=True)
            
        if all_pit_stops:
            merged_data['pit_stops'] = pd.concat(all_pit_stops, ignore_index=True)
            
        if all_results:
            merged_data['results'] = pd.concat(all_results, ignore_index=True)
        
        return merged_data
    
    def get_current_season_data(self):
        """
        Get current season data for prediction model.
        
        Returns:
            pandas.DataFrame: Current season standings and performance data
        """
        try:
            # Get current season race results
            current_year = 2024
            
            # Get all races completed this season
            schedule = ff1.get_event_schedule(current_year)
            completed_races = schedule[schedule['EventDate'] < pd.Timestamp.now()]
            
            all_results = []
            
            for _, race in completed_races.iterrows():
                try:
                    session = ff1.get_session(current_year, race['EventName'], 'R')
                    session.load()
                    results = session.results
                    results['RaceName'] = race['EventName']
                    results['Year'] = current_year
                    
                    # Map driver information - add Driver column from Abbreviation
                    results['Driver'] = results['Abbreviation']
                    results['Team'] = results['TeamName']
                    
                    all_results.append(results)
                except:
                    continue
            
            if all_results:
                return pd.concat(all_results, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching current season data: {e}")
            return pd.DataFrame()