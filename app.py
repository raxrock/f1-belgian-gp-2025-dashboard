import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
import random

# F1 team colors
F1_COLORS = {
    'Red Bull': '#0600EF',
    'Ferrari': '#DC143C',
    'Mercedes': '#00D2BE',
    'McLaren': '#FF8700',
    'Aston Martin': '#006F62',
    'Alpine': '#0090FF',
    'Williams': '#005AFF',
    'AlphaTauri': '#2B4562',
    'Alfa Romeo': '#900000',
    'Haas': '#FFFFFF'
}

# Tire compound colors
TIRE_COLORS = {
    'Soft': '#FF3333',
    'Medium': '#FFFF00',
    'Hard': '#FFFFFF',
    'Intermediate': '#00FF00',
    'Wet': '#0066FF'
}

def set_f1_theme():
    """Apply F1 official website-inspired theme to the Streamlit app"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap');
    
    .main {
        background: #15151e;
        color: #ffffff;
        font-family: 'Titillium Web', sans-serif;
    }
    
    .stApp {
        background: #15151e;
    }
    
    .stSidebar {
        background: #1a1a27;
        border-right: 1px solid #38383f;
    }
    
    .stSelectbox > div > div {
        background-color: #1a1a27;
        border: 1px solid #38383f;
        color: white;
        border-radius: 4px;
        font-family: 'Titillium Web', sans-serif;
    }
    
    .stButton > button {
        background: #e10600;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        font-family: 'Titillium Web', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
        padding: 12px 24px;
    }
    
    .stButton > button:hover {
        background: #c20500;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(225, 6, 0, 0.3);
    }
    
    .metric-card {
        background: #1a1a27;
        border: 1px solid #38383f;
        border-radius: 8px;
        padding: 20px;
        margin: 16px 0;
        font-family: 'Titillium Web', sans-serif;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #e10600;
        box-shadow: 0 4px 16px rgba(225, 6, 0, 0.1);
    }
    
    .title-header {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 900;
        font-family: 'Titillium Web', sans-serif;
        text-align: center;
        margin-bottom: 30px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .section-header {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'Titillium Web', sans-serif;
        margin: 30px 0 20px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-left: 4px solid #e10600;
        padding-left: 16px;
    }
    
    .f1-nav {
        background: #1a1a27;
        padding: 16px 0;
        border-bottom: 1px solid #38383f;
        margin-bottom: 30px;
    }
    
    .f1-hero {
        background: linear-gradient(135deg, #1a1a27 0%, #15151e 100%);
        padding: 40px 20px;
        border-radius: 8px;
        margin-bottom: 30px;
        border: 1px solid #38383f;
    }
    
    .f1-footer {
        background: #1a1a27;
        padding: 20px;
        border-top: 1px solid #38383f;
        margin-top: 40px;
        border-radius: 8px;
    }
    
    .legal-text {
        font-size: 0.8rem;
        color: #9999a3;
        line-height: 1.4;
        font-family: 'Titillium Web', sans-serif;
    }
    
    .stSlider > div > div > div > div {
        background-color: #e10600;
    }
    
    .stMultiSelect > div > div {
        background-color: #1a1a27;
        border: 1px solid #38383f;
        color: white;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Titillium Web', sans-serif;
        font-weight: 700;
    }
    
    .stMarkdown {
        font-family: 'Titillium Web', sans-serif;
    }
    
    .highlight-red {
        color: #e10600;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

def generate_mock_data():
    """Generate mock F1 Belgian GP data for demonstration"""
    drivers = [
        ('Max Verstappen', 'Red Bull', 0.35),
        ('Charles Leclerc', 'Ferrari', 0.18),
        ('Lewis Hamilton', 'Mercedes', 0.12),
        ('Sergio Perez', 'Red Bull', 0.10),
        ('Carlos Sainz', 'Ferrari', 0.08),
        ('George Russell', 'Mercedes', 0.07),
        ('Lando Norris', 'McLaren', 0.05),
        ('Fernando Alonso', 'Aston Martin', 0.03),
        ('Oscar Piastri', 'McLaren', 0.02)
    ]
    
    # Generate tire degradation data
    laps = np.arange(1, 45)
    degradation_data = {
        'Soft': 100 - (laps * 2.5) - (laps ** 1.2 * 0.1),
        'Medium': 100 - (laps * 1.8) - (laps ** 1.1 * 0.08),
        'Hard': 100 - (laps * 1.2) - (laps ** 1.05 * 0.05)
    }
    
    # Generate pit stop strategies
    strategies = {}
    for driver, team, _ in drivers:
        strategies[driver] = {
            'current': {
                'stops': 2,
                'compounds': ['Medium', 'Hard'],
                'pit_laps': [18, 35],
                'stint_lengths': [18, 17, 9],
                'predicted_time': f"{random.randint(86, 92)}:{random.randint(10, 59):02d}.{random.randint(100, 999)}"
            },
            'optimized': {
                'stops': 1,
                'compounds': ['Medium', 'Hard'],
                'pit_laps': [25],
                'stint_lengths': [25, 19],
                'predicted_time': f"{random.randint(84, 88)}:{random.randint(10, 59):02d}.{random.randint(100, 999)}"
            }
        }
    
    return drivers, degradation_data, strategies, laps

def create_winning_probabilities_chart(drivers_data):
    """Create interactive bar chart of winning probabilities"""
    drivers, teams, probabilities = zip(*drivers_data)
    colors = [F1_COLORS.get(team, '#FFFFFF') for team in teams]
    
    fig = go.Figure(data=[
        go.Bar(
            x=drivers,
            y=probabilities,
            marker_color=colors,
            marker_line_color='white',
            marker_line_width=2,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Win Probability: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '2025 Belgian GP Winner Predictions',
            'x': 0.5,
            'font': {'size': 20, 'color': '#FF1E1E'}
        },
        xaxis_title='Driver',
        yaxis_title='Win Probability',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#444'),
        yaxis=dict(gridcolor='#444', tickformat='.0%')
    )
    
    return fig

def create_tire_degradation_chart(degradation_data, laps):
    """Create tire degradation curves"""
    fig = go.Figure()
    
    for compound, performance in degradation_data.items():
        fig.add_trace(go.Scatter(
            x=laps,
            y=performance,
            mode='lines',
            name=compound,
            line=dict(color=TIRE_COLORS[compound], width=3),
            hovertemplate=f'<b>{compound}</b><br>Lap: %{{x}}<br>Performance: %{{y:.1f}}%<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'Tire Degradation Curves - Spa-Francorchamps',
            'x': 0.5,
            'font': {'size': 20, 'color': '#FF1E1E'}
        },
        xaxis_title='Lap Number',
        yaxis_title='Tire Performance (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#444'),
        yaxis=dict(gridcolor='#444'),
        legend=dict(bgcolor='rgba(0,0,0,0.5)')
    )
    
    return fig

def create_strategy_comparison_chart(driver_name, strategy_data):
    """Create pit stop strategy comparison chart"""
    current = strategy_data['current']
    optimized = strategy_data['optimized']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Current Strategy', 'Optimized Strategy'),
        vertical_spacing=0.1
    )
    
    # Current strategy
    for i, (stint, compound) in enumerate(zip(current['stint_lengths'], current['compounds'])):
        start_lap = sum(current['stint_lengths'][:i])
        fig.add_trace(go.Bar(
            x=[stint],
            y=[1],
            name=f'{compound} - {stint} laps',
            marker_color=TIRE_COLORS[compound],
            orientation='h',
            showlegend=False,
            hovertemplate=f'<b>{compound}</b><br>Stint Length: {stint} laps<extra></extra>'
        ), row=1, col=1)
    
    # Optimized strategy
    for i, (stint, compound) in enumerate(zip(optimized['stint_lengths'], optimized['compounds'])):
        start_lap = sum(optimized['stint_lengths'][:i])
        fig.add_trace(go.Bar(
            x=[stint],
            y=[1],
            name=f'{compound} - {stint} laps',
            marker_color=TIRE_COLORS[compound],
            orientation='h',
            showlegend=False,
            hovertemplate=f'<b>{compound}</b><br>Stint Length: {stint} laps<extra></extra>'
        ), row=2, col=1)
    
    fig.update_layout(
        title={
            'text': f'Pit Stop Strategy Comparison - {driver_name}',
            'x': 0.5,
            'font': {'size': 18, 'color': '#FF1E1E'}
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="F1 2025 Belgian GP Strategy Dashboard",
        page_icon="🏎️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    set_f1_theme()
    
    # Header with navigation style
    st.markdown("""
    <div class="f1-nav">
        <h1 class="title-header">🏎️ F1 2025 Belgian Grand Prix Strategy Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate mock data
    drivers_data, degradation_data, strategies, laps = generate_mock_data()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown('<h2 class="section-header">🏁 Race Control</h2>', unsafe_allow_html=True)
        
        # Driver/Team selection
        selected_driver = st.selectbox(
            "Select Driver for Detailed Analysis:",
            options=[driver for driver, _, _ in drivers_data],
            index=0
        )
        
        st.markdown("---")
        
        # Simulation parameters
        st.markdown('<h3 class="highlight-red">Simulation Parameters</h3>', unsafe_allow_html=True)
        
        num_stops = st.slider("Number of Pit Stops", 1, 3, 2)
        
        available_compounds = ['Soft', 'Medium', 'Hard']
        selected_compounds = st.multiselect(
            "Available Tire Compounds:",
            available_compounds,
            default=['Medium', 'Hard']
        )
        
        weather_condition = st.selectbox(
            "Weather Conditions:",
            ["Dry", "Light Rain", "Heavy Rain"],
            index=0
        )
        
        if st.button("🔄 Re-run Simulation"):
            st.success("Simulation updated with new parameters!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Winning probabilities chart
        st.markdown('<h2 class="section-header">🏆 Driver Championship Predictions</h2>', unsafe_allow_html=True)
        win_prob_chart = create_winning_probabilities_chart(drivers_data)
        st.plotly_chart(win_prob_chart, use_container_width=True)
        
        # Tire degradation chart
        st.markdown('<h2 class="section-header">🛞 Tire Performance Analysis</h2>', unsafe_allow_html=True)
        tire_chart = create_tire_degradation_chart(degradation_data, laps)
        st.plotly_chart(tire_chart, use_container_width=True)
        
        # Strategy comparison for selected driver
        st.markdown(f'<h2 class="section-header">⚡ Strategy Analysis - {selected_driver}</h2>', unsafe_allow_html=True)
        if selected_driver in strategies:
            strategy_chart = create_strategy_comparison_chart(selected_driver, strategies[selected_driver])
            st.plotly_chart(strategy_chart, use_container_width=True)
    
    with col2:
        # Key metrics and recommendations
        st.markdown('<h2 class="section-header">📊 Key Metrics</h2>', unsafe_allow_html=True)
        
        # Race winner prediction
        top_driver = max(drivers_data, key=lambda x: x[2])
        st.markdown(f"""
        <div class="metric-card">
            <h3>🥇 Predicted Winner</h3>
            <h2>{top_driver[0]}</h2>
            <p>Win Probability: {top_driver[2]:.1%}</p>
            <p>Team: {top_driver[1]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Strategy recommendations for selected driver
        if selected_driver in strategies:
            current_strategy = strategies[selected_driver]['current']
            optimized_strategy = strategies[selected_driver]['optimized']
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔧 Current Strategy</h3>
                <p><strong>Stops:</strong> {current_strategy['stops']}</p>
                <p><strong>Compounds:</strong> {', '.join(current_strategy['compounds'])}</p>
                <p><strong>Predicted Time:</strong> {current_strategy['predicted_time']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚡ Optimized Strategy</h3>
                <p><strong>Stops:</strong> {optimized_strategy['stops']}</p>
                <p><strong>Compounds:</strong> {', '.join(optimized_strategy['compounds'])}</p>
                <p><strong>Predicted Time:</strong> {optimized_strategy['predicted_time']}</p>
                <p style="color: #00FF00;"><strong>Time Gain:</strong> ~2.5 seconds</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Track information
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏁 Track Info</h3>
            <p><strong>Circuit:</strong> Spa-Francorchamps</p>
            <p><strong>Length:</strong> 7.004 km</p>
            <p><strong>Laps:</strong> 44</p>
            <p><strong>Weather:</strong> {weather_condition}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with legal disclaimers
    st.markdown("---")
    
    # Create expandable legal section
    with st.expander("📋 Legal Information & Disclaimers", expanded=False):
        st.markdown("""
        <div class="legal-text">
            <div style="background: #1a1a27; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h4 style="color: #e10600; margin-bottom: 15px;">⚠️ Legal Disclaimer</h4>
                <p>This dashboard is an independent project created for educational and demonstration purposes only. It is not affiliated with, endorsed by, or sponsored by Formula 1®, FIA, or any Formula 1® teams.</p>
            </div>
            
            <div style="background: #1a1a27; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h4 style="color: #e10600; margin-bottom: 15px;">🏛️ Trademarks & Copyrights</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <h5 style="color: #ffffff; margin-bottom: 10px;">Formula 1® Organizations:</h5>
                        <ul style="margin-left: 20px;">
                            <li>Formula 1®, F1® - Formula One Licensing B.V.</li>
                            <li>FIA® - Fédération Internationale de l'Automobile</li>
                            <li>Spa-Francorchamps® - Circuit de Spa-Francorchamps</li>
                        </ul>
                    </div>
                    <div>
                        <h5 style="color: #ffffff; margin-bottom: 10px;">Team Trademarks:</h5>
                        <ul style="margin-left: 20px;">
                            <li>Red Bull Racing® - Red Bull GmbH</li>
                            <li>Scuderia Ferrari® - Ferrari S.p.A.</li>
                            <li>Mercedes-AMG Petronas® - Mercedes-Benz Group AG</li>
                            <li>McLaren F1® - McLaren Group Limited</li>
                            <li>Aston Martin F1® - Aston Martin Lagonda Limited</li>
                        </ul>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <h5 style="color: #ffffff; margin-bottom: 10px;">Additional Teams:</h5>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <ul style="margin-left: 20px;">
                            <li>Alpine F1® - Renault Sport Racing</li>
                            <li>Williams Racing® - Williams Grand Prix Engineering</li>
                            <li>Visa Cash App RB® - Scuderia AlphaTauri S.p.A.</li>
                        </ul>
                        <ul style="margin-left: 20px;">
                            <li>Kick Sauber F1® - Sauber Motorsport AG</li>
                            <li>MoneyGram Haas F1® - Haas F1 Team</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div style="background: #1a1a27; padding: 20px; border-radius: 8px;">
                    <h4 style="color: #e10600; margin-bottom: 15px;">📊 Data & Analytics</h4>
                    <p>All race data, predictions, and analytics are simulated for demonstration purposes and do not represent actual Formula 1® race data, official predictions, or real team strategies.</p>
                </div>
                <div style="background: #1a1a27; padding: 20px; border-radius: 8px;">
                    <h4 style="color: #e10600; margin-bottom: 15px;">🚫 No Commercial Use</h4>
                    <p>This project is provided for educational purposes only. All rights to Formula 1® content, team identities, and racing data remain with their respective owners.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer credits
    st.markdown("""
    <div style="text-align: center; padding: 20px; margin-top: 30px; background: #1a1a27; border-radius: 8px;">
        <p style="color: #ffffff; font-size: 1.1rem; margin-bottom: 10px;">
            <strong>🏎️ F1 2025 Belgian GP Strategy Dashboard</strong>
        </p>
        <p style="color: #9999a3; font-size: 0.9rem; margin-bottom: 5px;">
            Built by <span style="color: #e10600; font-weight: 600;">Rakshith Kumar Karkala</span>
        </p>
        <p style="color: #9999a3; font-size: 0.8rem;">
            © 2025 - This project respects all intellectual property rights of Formula 1® and associated entities.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()