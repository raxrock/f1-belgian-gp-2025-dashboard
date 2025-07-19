"""
F1 Race Calendar Module
Provides functionality to display F1 race schedule with navigation controls
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from icalendar import Calendar
import pytz

def fetch_f1_calendar():
    """
    Fetch F1 race calendar from ICS URL and parse it into a DataFrame
    
    Returns:
        pd.DataFrame: DataFrame containing race schedule data
    """
    try:
        # F1 calendar ICS URL
        ics_url = "https://ics.ecal.com/ecal-sub/687907a42d850500089b44ef/Formula%201.ics"
        
        # Fetch the ICS file
        response = requests.get(ics_url, timeout=10)
        response.raise_for_status()
        
        # Parse the calendar
        calendar = Calendar.from_ical(response.content)
        
        races = []
        race_weekends = {}  # To group events by race weekend
        
        for component in calendar.walk():
            if component.name == "VEVENT":
                # Extract race information
                summary = str(component.get('summary', ''))
                location = str(component.get('location', ''))
                start_date = component.get('dtstart')
                end_date = component.get('dtend')
                
                # Parse dates
                if start_date and hasattr(start_date, 'dt'):
                    start_dt = start_date.dt
                    if isinstance(start_dt, datetime):
                        event_date = start_dt.date()
                    else:
                        event_date = start_dt
                else:
                    continue
                
                # Clean up race name by removing emojis and extracting GP name
                race_name = summary
                
                # Remove emojis and clean up the race name
                import re
                race_name = re.sub(r'[^\w\s\-\']', '', race_name)  # Remove emojis and special chars
                race_name = re.sub(r'\s+', ' ', race_name).strip()  # Normalize spaces
                
                # Extract meaningful race name
                if 'GRAND PRIX' in race_name.upper() or 'GRAN PREMIO' in race_name.upper():
                    # Look for country/location indicators
                    if 'BRITISH' in race_name.upper():
                        race_name = 'British Grand Prix'
                    elif 'BELGIAN' in race_name.upper() or 'BELGIUM' in race_name.upper():
                        race_name = 'Belgian Grand Prix'
                    elif 'ITALIAN' in race_name.upper() or 'ITALIA' in race_name.upper():
                        race_name = 'Italian Grand Prix'
                    elif 'HUNGARIAN' in race_name.upper():
                        race_name = 'Hungarian Grand Prix'
                    elif 'DUTCH' in race_name.upper() or 'NETHERLANDS' in race_name.upper():
                        race_name = 'Dutch Grand Prix'
                    elif 'MONACO' in race_name.upper():
                        race_name = 'Monaco Grand Prix'
                    elif 'SPANISH' in race_name.upper() or 'SPAIN' in race_name.upper():
                        race_name = 'Spanish Grand Prix'
                    elif 'CANADIAN' in race_name.upper() or 'CANADA' in race_name.upper():
                        race_name = 'Canadian Grand Prix'
                    elif 'AUSTRALIAN' in race_name.upper() or 'AUSTRALIA' in race_name.upper():
                        race_name = 'Australian Grand Prix'
                    elif 'JAPANESE' in race_name.upper() or 'JAPAN' in race_name.upper():
                        race_name = 'Japanese Grand Prix'
                    elif 'SINGAPORE' in race_name.upper():
                        race_name = 'Singapore Grand Prix'
                    elif 'MEXICAN' in race_name.upper() or 'MEXICO' in race_name.upper() or 'MÉXICO' in race_name.upper():
                        race_name = 'Mexican Grand Prix'
                    elif 'BRAZILIAN' in race_name.upper() or 'BRAZIL' in race_name.upper() or 'SAO PAULO' in race_name.upper() or 'PAULO' in race_name.upper():
                        race_name = 'Brazilian Grand Prix'
                    elif 'UNITED STATES' in race_name.upper() or 'USA' in race_name.upper():
                        race_name = 'United States Grand Prix'
                    elif 'ABU DHABI' in race_name.upper():
                        race_name = 'Abu Dhabi Grand Prix'
                    elif 'QATAR' in race_name.upper():
                        race_name = 'Qatar Grand Prix'
                    elif 'SAUDI' in race_name.upper():
                        race_name = 'Saudi Arabian Grand Prix'
                    elif 'BAHRAIN' in race_name.upper():
                        race_name = 'Bahrain Grand Prix'
                    elif 'AZERBAIJAN' in race_name.upper():
                        race_name = 'Azerbaijan Grand Prix'
                    elif 'MIAMI' in race_name.upper():
                        race_name = 'Miami Grand Prix'
                    elif 'LAS VEGAS' in race_name.upper():
                        race_name = 'Las Vegas Grand Prix'
                    else:
                        # Fallback: extract country name from location
                        if location:
                            country_name = location.split(',')[-1].strip()
                            race_name = f"{country_name} Grand Prix"
                        else:
                            race_name = "Formula 1 Grand Prix"
                
                # Only process race events (not practice or qualifying)
                if ('🏁' in summary or 'RACE' in summary.upper()) and 'PRACTICE' not in summary.upper() and 'QUALIFYING' not in summary.upper():
                    # Extract circuit and country from location
                    circuit = "TBD"
                    country = "TBD"
                    
                    # Map race names to proper circuit names
                    circuit_mapping = {
                        'British Grand Prix': 'Silverstone Circuit',
                        'Belgian Grand Prix': 'Circuit de Spa-Francorchamps',
                        'Hungarian Grand Prix': 'Hungaroring',
                        'Dutch Grand Prix': 'Circuit Zandvoort',
                        'Italian Grand Prix': 'Autodromo Nazionale Monza',
                        'Singapore Grand Prix': 'Marina Bay Street Circuit',
                        'United States Grand Prix': 'Circuit of the Americas',
                        'Mexican Grand Prix': 'Autódromo Hermanos Rodríguez',
                        'Brazilian Grand Prix': 'Autódromo José Carlos Pace',
                        'Las Vegas Grand Prix': 'Las Vegas Street Circuit',
                        'Qatar Grand Prix': 'Lusail International Circuit',
                        'Abu Dhabi Grand Prix': 'Yas Marina Circuit'
                    }
                    
                    country_mapping = {
                        'British Grand Prix': 'United Kingdom',
                        'Belgian Grand Prix': 'Belgium',
                        'Hungarian Grand Prix': 'Hungary',
                        'Dutch Grand Prix': 'Netherlands',
                        'Italian Grand Prix': 'Italy',
                        'Singapore Grand Prix': 'Singapore',
                        'United States Grand Prix': 'United States',
                        'Mexican Grand Prix': 'Mexico',
                        'Brazilian Grand Prix': 'Brazil',
                        'Las Vegas Grand Prix': 'United States',
                        'Qatar Grand Prix': 'Qatar',
                        'Abu Dhabi Grand Prix': 'United Arab Emirates'
                    }
                    
                    # Get circuit and country from mapping
                    circuit = circuit_mapping.get(race_name, location if location else "TBD")
                    country = country_mapping.get(race_name, "TBD")
                    
                    # Fallback to parsing location if mapping fails
                    if circuit == "TBD" and location:
                        if ',' in location:
                            parts = location.split(',')
                            circuit = parts[0].strip()
                            country = parts[-1].strip()
                        else:
                            circuit = location
                            country = location
                    
                    # Group by race weekend (normalize race key)
                    race_key = race_name.strip()
                    
                    # Additional normalization for grouping
                    if 'GRANDE PRÊMIO' in race_key or 'PAULO' in race_key:
                        race_key = 'Brazilian Grand Prix'
                        race_name = 'Brazilian Grand Prix'
                    
                    if race_key not in race_weekends:
                        race_weekends[race_key] = {
                            'race_name': race_name,
                            'circuit': circuit,
                            'country': country,
                            'race_date': event_date,
                            'weekend_start': event_date,
                            'weekend_end': event_date
                        }
                    else:
                        # Update weekend dates
                        if event_date < race_weekends[race_key]['weekend_start']:
                            race_weekends[race_key]['weekend_start'] = event_date
                        if event_date > race_weekends[race_key]['weekend_end']:
                            race_weekends[race_key]['weekend_end'] = event_date
                            race_weekends[race_key]['race_date'] = event_date  # Race is typically the last event
        
        # Convert to list
        races = list(race_weekends.values())
        
        # Convert to DataFrame and sort by race date
        df = pd.DataFrame(races)
        if not df.empty:
            df = df.sort_values('race_date').reset_index(drop=True)
            
            # Add formatted weekend dates
            df['weekend_dates'] = df.apply(
                lambda row: f"{row['weekend_start'].strftime('%b %d')} - {row['weekend_end'].strftime('%b %d, %Y')}" 
                if row['weekend_start'] != row['weekend_end'] 
                else row['race_date'].strftime('%b %d, %Y'), 
                axis=1
            )
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching F1 calendar: {str(e)}")
        # Return fallback data
        return get_fallback_calendar()

def get_fallback_calendar():
    """
    Fallback race calendar data in case ICS fetch fails
    
    Returns:
        pd.DataFrame: DataFrame with fallback race schedule
    """
    fallback_races = [
        {
            'race_name': 'British Grand Prix',
            'circuit': 'Silverstone Circuit',
            'country': 'United Kingdom',
            'race_date': datetime(2025, 7, 6).date(),
            'end_date': datetime(2025, 7, 6).date(),
            'weekend_start': datetime(2025, 7, 4).date(),
            'weekend_end': datetime(2025, 7, 6).date(),
            'weekend_dates': 'Jul 04 - Jul 06, 2025'
        },
        {
            'race_name': 'Belgian Grand Prix',
            'circuit': 'Circuit de Spa-Francorchamps',
            'country': 'Belgium',
            'race_date': datetime(2025, 7, 27).date(),
            'end_date': datetime(2025, 7, 27).date(),
            'weekend_start': datetime(2025, 7, 25).date(),
            'weekend_end': datetime(2025, 7, 27).date(),
            'weekend_dates': 'Jul 25 - Jul 27, 2025'
        },
        {
            'race_name': 'Hungarian Grand Prix',
            'circuit': 'Hungaroring',
            'country': 'Hungary',
            'race_date': datetime(2025, 8, 3).date(),
            'end_date': datetime(2025, 8, 3).date(),
            'weekend_start': datetime(2025, 8, 1).date(),
            'weekend_end': datetime(2025, 8, 3).date(),
            'weekend_dates': 'Aug 01 - Aug 03, 2025'
        },
        {
            'race_name': 'Dutch Grand Prix',
            'circuit': 'Circuit Zandvoort',
            'country': 'Netherlands',
            'race_date': datetime(2025, 8, 31).date(),
            'end_date': datetime(2025, 8, 31).date(),
            'weekend_start': datetime(2025, 8, 29).date(),
            'weekend_end': datetime(2025, 8, 31).date(),
            'weekend_dates': 'Aug 29 - Aug 31, 2025'
        },
        {
            'race_name': 'Italian Grand Prix',
            'circuit': 'Autodromo Nazionale di Monza',
            'country': 'Italy',
            'race_date': datetime(2025, 9, 7).date(),
            'end_date': datetime(2025, 9, 7).date(),
            'weekend_start': datetime(2025, 9, 5).date(),
            'weekend_end': datetime(2025, 9, 7).date(),
            'weekend_dates': 'Sep 05 - Sep 07, 2025'
        }
    ]
    
    return pd.DataFrame(fallback_races)

def find_next_race_index(df):
    """
    Find the index of the next upcoming race based on current date
    
    Args:
        df (pd.DataFrame): Race schedule DataFrame
        
    Returns:
        int: Index of the next race, or 0 if no upcoming races
    """
    today = datetime.now().date()
    
    # Find the first race that hasn't happened yet
    for idx, row in df.iterrows():
        if row['race_date'] >= today:
            return idx
    
    # If no upcoming races, return the last race
    return len(df) - 1

def race_calendar():
    """
    Display F1 Race Calendar with navigation controls
    
    This function creates an interactive race calendar widget that shows
    race information with Previous/Next navigation buttons.
    """
    # Initialize session state for current race index
    if 'current_race_index' not in st.session_state:
        st.session_state.current_race_index = None
    
    # Fetch race calendar data
    with st.spinner("Loading F1 race calendar..."):
        df = fetch_f1_calendar()
    
    if df.empty:
        st.error("No race data available")
        return
    
    # Set initial race index to next upcoming race
    if st.session_state.current_race_index is None:
        st.session_state.current_race_index = find_next_race_index(df)
    
    # Ensure current_race_index is within bounds
    if st.session_state.current_race_index >= len(df):
        st.session_state.current_race_index = len(df) - 1
    elif st.session_state.current_race_index < 0:
        st.session_state.current_race_index = 0
    
    # Get current race data
    current_race = df.iloc[st.session_state.current_race_index]
    
    # Create full-width container with light background
    with st.container():
        # Add horizontal divider
        st.markdown("---")
        
        # Calendar header with responsive design
        st.markdown("""
        <style>
        .f1-calendar-container {
            background: linear-gradient(135deg, #1a1a27 0%, #2d2d3a 100%);
            border-radius: 12px;
            padding: 40px;
            margin: 20px 0;
            border: 2px solid #e10600;
            box-shadow: 0 8px 32px rgba(225, 6, 0, 0.2);
        }
        
        @media (max-width: 768px) {
            .f1-calendar-container {
                padding: 20px;
                margin: 10px 0;
            }
        }
        
        .f1-calendar-title {
            color: #ffffff;
            text-align: center;
            margin-bottom: 30px;
            font-family: 'Titillium Web', sans-serif;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-size: 2.2rem;
            text-shadow: 0 2px 4px rgba(225, 6, 0, 0.3);
        }
        
        @media (max-width: 768px) {
            .f1-calendar-title {
                font-size: 1.8rem;
                letter-spacing: 1px;
                margin-bottom: 20px;
            }
        }
        
        .f1-race-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(225, 6, 0, 0.2);
            border: 3px solid #e10600;
            margin: 0 20px;
        }
        
        @media (max-width: 768px) {
            .f1-race-card {
                margin: 0 10px;
                padding: 20px;
            }
        }
        </style>
        
        <div class="f1-calendar-container">
        """, unsafe_allow_html=True)
        
        # Title
        st.markdown("""
        <h2 class="f1-calendar-title">
            🏁 F1 2025 Race Calendar
        </h2>
        """, unsafe_allow_html=True)
        
        # Navigation buttons and race info
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            # Previous race button
            if st.button(
                "← Previous Race",
                disabled=(st.session_state.current_race_index == 0),
                key="prev_race",
                help="Go to previous race" if st.session_state.current_race_index > 0 else "No previous race"
            ):
                if st.session_state.current_race_index > 0:
                    st.session_state.current_race_index -= 1
                    st.rerun()
        
        with col2:
            # Race information card using Streamlit components
            st.markdown('<div class="f1-race-card">', unsafe_allow_html=True)
            
            # Race name
            st.markdown(f"""
            <h2 style="
                color: #e10600;
                margin: 0 0 20px 0;
                font-size: 1.8rem;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-family: 'Titillium Web', sans-serif;
            ">{current_race['race_name']}</h2>
            """, unsafe_allow_html=True)
            
            # Circuit
            st.markdown(f"""
            <div style="
                background: rgba(225, 6, 0, 0.1);
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
                border-left: 4px solid #e10600;
            ">
                <p style="
                    color: #2c3e50;
                    margin: 0;
                    font-size: 1.2rem;
                    font-weight: 700;
                    font-family: 'Titillium Web', sans-serif;
                ">🏎️ {current_race['circuit']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Country
            st.markdown(f"""
            <div style="
                background: rgba(108, 117, 125, 0.1);
                border-radius: 8px;
                padding: 12px;
                margin: 15px 0;
            ">
                <p style="
                    color: #495057;
                    margin: 0;
                    font-size: 1.1rem;
                    font-weight: 600;
                    font-family: 'Titillium Web', sans-serif;
                ">📍 {current_race['country']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Race dates
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #e10600 0%, #c20500 100%);
                border-radius: 8px;
                padding: 15px;
                margin: 20px 0 0 0;
            ">
                <p style="
                    color: white;
                    margin: 0;
                    font-size: 1.1rem;
                    font-weight: 600;
                    font-family: 'Titillium Web', sans-serif;
                ">📅 {current_race['weekend_dates']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Close the main container
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            # Next race button
            if st.button(
                "Next Race →",
                disabled=(st.session_state.current_race_index == len(df) - 1),
                key="next_race",
                help="Go to next race" if st.session_state.current_race_index < len(df) - 1 else "No next race"
            ):
                if st.session_state.current_race_index < len(df) - 1:
                    st.session_state.current_race_index += 1
                    st.rerun()
        
        # Race counter
        st.markdown(f"""
        <div style="text-align: center; margin-top: 25px;">
            <p style="
                color: #9999a3;
                font-size: 1rem;
                margin: 0;
                font-family: 'Titillium Web', sans-serif;
                font-weight: 600;
                letter-spacing: 0.5px;
            ">
                Race {st.session_state.current_race_index + 1} of {len(df)}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Close the calendar container div
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add bottom spacing
        st.markdown("<br>", unsafe_allow_html=True)