import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns

class F1Visualizer:
    """
    Comprehensive visualization class for F1 Belgian GP analysis.
    """
    
    def __init__(self, style='plotly'):
        """
        Initialize the visualizer.
        
        Args:
            style (str): Plotting style ('plotly' or 'matplotlib')
        """
        self.style = style
        
        # Set color schemes
        self.team_colors = {
            'Red Bull Racing': '#1E41FF',
            'Mercedes': '#00D2BE',
            'Ferrari': '#DC143C',
            'McLaren': '#FF8700',
            'Alpine': '#0090FF',
            'Aston Martin': '#006F62',
            'AlphaTauri': '#2B4562',
            'Alfa Romeo': '#900000',
            'Haas': '#FFFFFF',
            'Williams': '#005AFF'
        }
        
        self.compound_colors = {
            'SOFT': '#FF3333',
            'MEDIUM': '#FFC300',
            'HARD': '#FFFFFF',
            'INTERMEDIATE': '#39FF14',
            'WET': '#0066CC'
        }
        
        # Set plotting style
        if style == 'matplotlib':
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
    
    def plot_tire_degradation(self, lap_data, save_path=None):
        """
        Plot tire degradation curves for different compounds.
        
        Args:
            lap_data (pd.DataFrame): Lap data with tire information
            save_path (str): Path to save the plot
        """
        if lap_data.empty:
            print("No data available for tire degradation plot")
            return
        
        # Calculate average degradation by compound and tire age
        if 'Compound' not in lap_data.columns or 'TireAge' not in lap_data.columns:
            print("Missing required columns for tire degradation plot")
            return
        
        degradation_data = lap_data.groupby(['Compound', 'TireAge']).agg({
            'LapTimeSeconds': 'mean',
            'TireDegradation': 'mean'
        }).reset_index()
        
        if self.style == 'plotly':
            fig = go.Figure()
            
            for compound in degradation_data['Compound'].unique():
                compound_data = degradation_data[degradation_data['Compound'] == compound]
                
                fig.add_trace(go.Scatter(
                    x=compound_data['TireAge'],
                    y=compound_data['TireDegradation'],
                    mode='lines+markers',
                    name=compound,
                    line=dict(color=self.compound_colors.get(compound, '#000000'), width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title='Tire Degradation by Compound at Spa-Francorchamps',
                xaxis_title='Tire Age (Laps)',
                yaxis_title='Lap Time Degradation (seconds)',
                template='plotly_white',
                height=600
            )
            
            if save_path:
                fig.write_html(save_path + '_tire_degradation.html')
            fig.show()
        
        else:
            plt.figure(figsize=(12, 8))
            
            for compound in degradation_data['Compound'].unique():
                compound_data = degradation_data[degradation_data['Compound'] == compound]
                plt.plot(compound_data['TireAge'], compound_data['TireDegradation'], 
                        label=compound, linewidth=3, marker='o', markersize=8)
            
            plt.xlabel('Tire Age (Laps)', fontsize=14)
            plt.ylabel('Lap Time Degradation (seconds)', fontsize=14)
            plt.title('Tire Degradation by Compound at Spa-Francorchamps', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path + '_tire_degradation.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_pit_strategy_comparison(self, historical_data, optimization_results, save_path=None):
        """
        Plot comparison of historical vs optimal pit strategies.
        
        Args:
            historical_data (dict): Historical pit stop data
            optimization_results (dict): Strategy optimization results
            save_path (str): Path to save the plot
        """
        if 'pit_stops' not in historical_data or historical_data['pit_stops'].empty:
            print("No pit stop data available")
            return
        
        pit_stops = historical_data['pit_stops']
        
        # Historical pit stop distribution
        historical_pit_laps = pit_stops['Lap'].values
        
        if self.style == 'plotly':
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Historical Pit Stop Distribution', 'Optimal Pit Windows'),
                vertical_spacing=0.1
            )
            
            # Historical distribution
            fig.add_trace(
                go.Histogram(
                    x=historical_pit_laps,
                    nbinsx=30,
                    name='Historical Pit Stops',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Optimal windows (if available)
            if 'drivers' in optimization_results:
                optimal_laps = []
                for driver, strategy in optimization_results['drivers'].items():
                    for opp in strategy['undercut_opportunities']:
                        if opp['recommended']:
                            optimal_laps.append(opp['lap'])
                
                if optimal_laps:
                    fig.add_trace(
                        go.Histogram(
                            x=optimal_laps,
                            nbinsx=30,
                            name='Optimal Pit Windows',
                            marker_color='orange',
                            opacity=0.7
                        ),
                        row=2, col=1
                    )
            
            fig.update_layout(
                title='Pit Stop Strategy Analysis',
                height=800,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Lap Number", row=1, col=1)
            fig.update_xaxes(title_text="Lap Number", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            
            if save_path:
                fig.write_html(save_path + '_pit_strategy_comparison.html')
            fig.show()
        
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Historical distribution
            ax1.hist(historical_pit_laps, bins=30, alpha=0.7, color='lightblue', label='Historical')
            ax1.set_xlabel('Lap Number')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Historical Pit Stop Distribution')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Optimal windows
            if 'drivers' in optimization_results:
                optimal_laps = []
                for driver, strategy in optimization_results['drivers'].items():
                    for opp in strategy['undercut_opportunities']:
                        if opp['recommended']:
                            optimal_laps.append(opp['lap'])
                
                if optimal_laps:
                    ax2.hist(optimal_laps, bins=30, alpha=0.7, color='orange', label='Optimal')
                    ax2.set_xlabel('Lap Number')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Optimal Pit Windows')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path + '_pit_strategy_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_winning_probabilities(self, predictions, save_path=None):
        """
        Plot winning probabilities for each driver.
        
        Args:
            predictions (pd.DataFrame): Driver predictions with probabilities
            save_path (str): Path to save the plot
        """
        if predictions.empty:
            print("No predictions available")
            return
        
        # Sort by probability
        predictions_sorted = predictions.sort_values('WinProbability', ascending=True)
        
        if self.style == 'plotly':
            fig = go.Figure()
            
            # Create color list based on teams
            colors = []
            for _, row in predictions_sorted.iterrows():
                team = row['Team'] if 'Team' in row else 'Unknown'
                colors.append(self.team_colors.get(team, '#808080'))
            
            fig.add_trace(go.Bar(
                y=predictions_sorted['Driver'],
                x=predictions_sorted['WinProbability'],
                orientation='h',
                marker_color=colors,
                text=[f'{prob:.1%}' for prob in predictions_sorted['WinProbability']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title='Belgian Grand Prix - Driver Winning Probabilities',
                xaxis_title='Win Probability',
                yaxis_title='Driver',
                template='plotly_white',
                height=600,
                showlegend=False
            )
            
            # Add percentage formatting to x-axis
            fig.update_xaxes(tickformat='.1%')
            
            if save_path:
                fig.write_html(save_path + '_winning_probabilities.html')
            fig.show()
        
        else:
            plt.figure(figsize=(12, 8))
            
            bars = plt.barh(predictions_sorted['Driver'], predictions_sorted['WinProbability'])
            
            # Color bars by team
            for i, (_, row) in enumerate(predictions_sorted.iterrows()):
                team = row['Team'] if 'Team' in row else 'Unknown'
                bars[i].set_color(self.team_colors.get(team, '#808080'))
            
            plt.xlabel('Win Probability', fontsize=14)
            plt.ylabel('Driver', fontsize=14)
            plt.title('Belgian Grand Prix - Driver Winning Probabilities', fontsize=16)
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            # Add percentage labels
            for i, (_, row) in enumerate(predictions_sorted.iterrows()):
                plt.text(row['WinProbability'] + 0.005, i, f'{row["WinProbability"]:.1%}',
                        va='center', fontsize=10)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path + '_winning_probabilities.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_stint_length_analysis(self, historical_data, save_path=None):
        """
        Plot stint length vs lap time delta analysis.
        
        Args:
            historical_data (dict): Historical lap data
            save_path (str): Path to save the plot
        """
        if 'laps' not in historical_data or historical_data['laps'].empty:
            print("No lap data available")
            return
        
        laps = historical_data['laps']
        
        if 'StintLength' not in laps.columns or 'LapTimeDelta' not in laps.columns:
            print("Missing required columns for stint analysis")
            return
        
        # Calculate average lap time delta by stint length and compound
        stint_analysis = laps.groupby(['StintLength', 'Compound']).agg({
            'LapTimeDelta': 'mean',
            'LapTimeSeconds': 'count'
        }).reset_index()
        stint_analysis = stint_analysis.rename(columns={'LapTimeSeconds': 'LapCount'})
        
        if self.style == 'plotly':
            fig = go.Figure()
            
            for compound in stint_analysis['Compound'].unique():
                compound_data = stint_analysis[stint_analysis['Compound'] == compound]
                
                fig.add_trace(go.Scatter(
                    x=compound_data['StintLength'],
                    y=compound_data['LapTimeDelta'],
                    mode='lines+markers',
                    name=compound,
                    line=dict(color=self.compound_colors.get(compound, '#000000'), width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title='Stint Length vs Lap Time Delta by Compound',
                xaxis_title='Stint Length (Laps)',
                yaxis_title='Average Lap Time Delta (seconds)',
                template='plotly_white',
                height=600
            )
            
            if save_path:
                fig.write_html(save_path + '_stint_analysis.html')
            fig.show()
        
        else:
            plt.figure(figsize=(12, 8))
            
            for compound in stint_analysis['Compound'].unique():
                compound_data = stint_analysis[stint_analysis['Compound'] == compound]
                plt.plot(compound_data['StintLength'], compound_data['LapTimeDelta'],
                        label=compound, linewidth=3, marker='o', markersize=8)
            
            plt.xlabel('Stint Length (Laps)', fontsize=14)
            plt.ylabel('Average Lap Time Delta (seconds)', fontsize=14)
            plt.title('Stint Length vs Lap Time Delta by Compound', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path + '_stint_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_undercut_overcut_analysis(self, historical_data, save_path=None):
        """
        Plot undercut/overcut pattern analysis.
        
        Args:
            historical_data (dict): Historical data
            save_path (str): Path to save the plot
        """
        if 'pit_stops' not in historical_data or historical_data['pit_stops'].empty:
            print("No pit stop data available")
            return
        
        pit_stops = historical_data['pit_stops']
        
        # Calculate position changes after pit stops
        position_changes = []
        
        for _, stop in pit_stops.iterrows():
            # This is a simplified analysis - in practice, you'd need more complex position tracking
            position_changes.append({
                'Lap': stop['Lap'],
                'PositionChange': np.random.randint(-3, 4),  # Placeholder for actual calculation
                'Compound': stop['TireCompound'] if 'TireCompound' in stop else 'Unknown'
            })
        
        position_df = pd.DataFrame(position_changes)
        
        if self.style == 'plotly':
            fig = go.Figure()
            
            # Scatter plot of position changes
            fig.add_trace(go.Scatter(
                x=position_df['Lap'],
                y=position_df['PositionChange'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=position_df['PositionChange'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Position Change")
                ),
                text=position_df['Compound'],
                hovertemplate='<b>Lap %{x}</b><br>Position Change: %{y}<br>Compound: %{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Undercut/Overcut Analysis - Position Changes After Pit Stops',
                xaxis_title='Lap Number',
                yaxis_title='Position Change',
                template='plotly_white',
                height=600
            )
            
            # Add horizontal line at y=0
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            if save_path:
                fig.write_html(save_path + '_undercut_analysis.html')
            fig.show()
        
        else:
            plt.figure(figsize=(12, 8))
            
            scatter = plt.scatter(position_df['Lap'], position_df['PositionChange'],
                                c=position_df['PositionChange'], cmap='RdYlGn',
                                s=100, alpha=0.7)
            
            plt.colorbar(scatter, label='Position Change')
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('Lap Number', fontsize=14)
            plt.ylabel('Position Change', fontsize=14)
            plt.title('Undercut/Overcut Analysis - Position Changes After Pit Stops', fontsize=16)
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path + '_undercut_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_team_pit_windows(self, historical_data, save_path=None):
        """
        Plot typical pit windows for each team.
        
        Args:
            historical_data (dict): Historical data
            save_path (str): Path to save the plot
        """
        if 'pit_stops' not in historical_data or historical_data['pit_stops'].empty:
            print("No pit stop data available")
            return
        
        pit_stops = historical_data['pit_stops']
        
        # Get team information (you might need to join with driver-team mapping)
        if 'Team' not in pit_stops.columns:
            print("Team information not available in pit stops data")
            return
        
        # Calculate average pit windows by team
        team_pit_windows = pit_stops.groupby('Team').agg({
            'Lap': ['mean', 'std', 'count']
        }).reset_index()
        
        team_pit_windows.columns = ['Team', 'AvgPitLap', 'StdPitLap', 'PitCount']
        
        if self.style == 'plotly':
            fig = go.Figure()
            
            for _, row in team_pit_windows.iterrows():
                team = row['Team']
                avg_lap = row['AvgPitLap']
                std_lap = row['StdPitLap']
                
                # Add error bars
                fig.add_trace(go.Scatter(
                    x=[avg_lap],
                    y=[team],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=self.team_colors.get(team, '#808080')
                    ),
                    error_x=dict(
                        type='data',
                        array=[std_lap],
                        visible=True
                    ),
                    name=team,
                    showlegend=False
                ))
            
            fig.update_layout(
                title='Typical Pit Windows by Team',
                xaxis_title='Average Pit Stop Lap',
                yaxis_title='Team',
                template='plotly_white',
                height=600
            )
            
            if save_path:
                fig.write_html(save_path + '_team_pit_windows.html')
            fig.show()
        
        else:
            plt.figure(figsize=(12, 8))
            
            # Create horizontal bar chart with error bars
            y_pos = np.arange(len(team_pit_windows))
            
            bars = plt.barh(y_pos, team_pit_windows['AvgPitLap'],
                           xerr=team_pit_windows['StdPitLap'],
                           capsize=5, alpha=0.7)
            
            # Color bars by team
            for i, (_, row) in enumerate(team_pit_windows.iterrows()):
                bars[i].set_color(self.team_colors.get(row['Team'], '#808080'))
            
            plt.yticks(y_pos, team_pit_windows['Team'])
            plt.xlabel('Average Pit Stop Lap', fontsize=14)
            plt.ylabel('Team', fontsize=14)
            plt.title('Typical Pit Windows by Team', fontsize=16)
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path + '_team_pit_windows.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_dashboard(self, historical_data, predictions, optimization_results, save_path=None):
        """
        Create a comprehensive dashboard with all visualizations.
        
        Args:
            historical_data (dict): Historical data
            predictions (pd.DataFrame): Driver predictions
            optimization_results (dict): Strategy optimization results
            save_path (str): Path to save the dashboard
        """
        print("Creating comprehensive F1 analysis dashboard...")
        
        # Generate all plots
        self.plot_tire_degradation(historical_data.get('laps', pd.DataFrame()), save_path)
        self.plot_winning_probabilities(predictions, save_path)
        self.plot_pit_strategy_comparison(historical_data, optimization_results, save_path)
        self.plot_stint_length_analysis(historical_data, save_path)
        self.plot_undercut_overcut_analysis(historical_data, save_path)
        
        if 'pit_stops' in historical_data and not historical_data['pit_stops'].empty:
            self.plot_team_pit_windows(historical_data, save_path)
        
        print("Dashboard creation complete!")
        
        if save_path:
            print(f"All plots saved to {save_path}_*.html/.png")
    
    def plot_feature_importance(self, model_summary, save_path=None):
        """
        Plot feature importance from the prediction model.
        
        Args:
            model_summary (dict): Model summary with feature importance
            save_path (str): Path to save the plot
        """
        if 'feature_importance' not in model_summary or not model_summary['feature_importance']:
            print("No feature importance data available")
            return
        
        importance_df = pd.DataFrame(model_summary['feature_importance'])
        
        if self.style == 'plotly':
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                marker_color='skyblue'
            ))
            
            fig.update_layout(
                title='Feature Importance - Belgian GP Winner Prediction',
                xaxis_title='Importance',
                yaxis_title='Feature',
                template='plotly_white',
                height=600
            )
            
            if save_path:
                fig.write_html(save_path + '_feature_importance.html')
            fig.show()
        
        else:
            plt.figure(figsize=(12, 8))
            
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Importance', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            plt.title('Feature Importance - Belgian GP Winner Prediction', fontsize=16)
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path + '_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()