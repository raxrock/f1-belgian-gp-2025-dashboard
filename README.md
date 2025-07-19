# F1 2025 Belgian Grand Prix Strategy Dashboard

An interactive web dashboard showcasing F1 2025 Belgian Grand Prix winner predictions and pit stop strategy optimization using Streamlit.

**Built by: Rakshith Kumar Karkala**

## Features

- **Driver Win Probability Visualization**: Interactive bar chart showing predicted winning chances for each driver
- **Tire Degradation Analysis**: Real-time tire performance curves for different compounds
- **Pit Stop Strategy Comparison**: Side-by-side comparison of current vs optimized strategies
- **Interactive Controls**: Adjust simulation parameters and re-run analyses
- **F1 Race Calendar**: Live race schedule with navigation controls, fetched from official F1 calendar
- **F1-Inspired Theme**: Racing-style UI with official F1 team colors and sleek design

## Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you have the following files in your directory:
   # - app.py
   # - dashboard_requirements.txt
   # - README.md
   ```

2. **Install dependencies**
   ```bash
   pip install -r dashboard_requirements.txt
   ```

## Running the App

1. **Start the Streamlit server**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   The app will automatically open in your default browser at `http://localhost:8501`

## Usage

### Dashboard Sections

1. **Race Control Sidebar**
   - Select any driver for detailed strategy analysis
   - Adjust simulation parameters (pit stops, tire compounds, weather)
   - Re-run simulations with new parameters

2. **Main Dashboard**
   - **Driver Championship Predictions**: Bar chart of win probabilities
   - **Tire Performance Analysis**: Degradation curves for all compounds
   - **Strategy Analysis**: Detailed comparison for selected driver

3. **Key Metrics Panel**
   - Predicted race winner with probability
   - Current vs optimized strategy comparison
   - Track information and conditions

### Interactive Features

- **Hover over charts** for detailed information
- **Select different drivers** to view their specific strategies
- **Adjust simulation parameters** to see how changes affect predictions
- **Compare strategies** to identify optimization opportunities

## Technical Details

- **Framework**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Theme**: Custom F1-inspired CSS styling
- **Data**: Mock simulation data for demonstration

## Dependencies

- `streamlit>=1.28.0` - Web app framework
- `plotly>=5.15.0` - Interactive visualizations
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computations
- `requests>=2.31.0` - HTTP requests for calendar data
- `icalendar>=5.0.0` - ICS calendar file parsing
- `pytz>=2023.3` - Timezone handling

## Customization

To use real F1 data instead of mock data:
1. Replace the `generate_mock_data()` function with your actual data loading logic
2. Ensure data follows the same structure as the mock data
3. Update visualization functions if needed for your specific data format

## Troubleshooting

- **Port already in use**: Use `streamlit run app.py --server.port 8502` to run on a different port
- **Module not found**: Ensure all dependencies are installed with `pip install -r dashboard_requirements.txt`
- **Slow loading**: The app generates mock data on startup; real data integration may improve performance

## Legal Disclaimer

This dashboard is an independent project created for educational and demonstration purposes only. It is not affiliated with, endorsed by, or sponsored by Formula 1®, FIA, or any Formula 1® teams.

### Trademarks & Copyrights

- Formula 1®, F1®, and related marks are trademarks of Formula One Licensing B.V.
- All team names, logos, and colors are trademarks of their respective Formula 1® teams:
  - Red Bull Racing® is a trademark of Red Bull GmbH
  - Scuderia Ferrari® is a trademark of Ferrari S.p.A.
  - Mercedes-AMG Petronas F1 Team® is a trademark of Mercedes-Benz Group AG
  - McLaren F1 Team® is a trademark of McLaren Group Limited
  - Aston Martin F1 Team® is a trademark of Aston Martin Lagonda Limited
  - Alpine F1 Team® is a trademark of Renault Sport Racing
  - Williams Racing® is a trademark of Williams Grand Prix Engineering Limited
  - Visa Cash App RB Formula One Team® is a trademark of Scuderia AlphaTauri S.p.A.
  - Kick Sauber F1 Team® is a trademark of Sauber Motorsport AG
  - MoneyGram Haas F1 Team® is a trademark of Haas F1 Team
- FIA® is a trademark of the Fédération Internationale de l'Automobile
- Circuit de Spa-Francorchamps® is a trademark of Circuit de Spa-Francorchamps

### Data & Analytics

All race data, predictions, and analytics presented in this dashboard are simulated for demonstration purposes and do not represent actual Formula 1® race data, official predictions, or real team strategies.

### No Commercial Use

This project is provided for educational purposes only and is not intended for commercial use. All rights to Formula 1® content, team identities, and racing data remain with their respective owners.

---

© 2025 Rakshith Kumar Karkala. This project respects all intellectual property rights of Formula 1® and associated entities.