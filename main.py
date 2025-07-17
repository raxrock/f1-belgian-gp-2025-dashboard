#!/usr/bin/env python3
"""
Belgian Grand Prix Winner Prediction and Pit Stop Strategy Optimizer

This script performs a comprehensive analysis of the Belgian Grand Prix using historical
F1 data to predict race winners and optimize pit stop strategies.

Features:
- Historical data fetching and processing
- Tire degradation analysis
- Pit stop strategy optimization using Monte Carlo simulation
- Race winner prediction using machine learning
- Comprehensive visualizations and reporting

Usage:
    python main.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Import our modules
from data_loader import BelgianGPDataLoader
from features import FeatureEngineer
from strategy_optimizer import PitStopOptimizer
from model import BelgianGPPredictor
from visualize import F1Visualizer

def main():
    """
    Main execution function for the Belgian GP analysis pipeline.
    """
    print("=" * 60)
    print("BELGIAN GRAND PRIX ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    YEARS = list(range(2018, 2025))  # Historical data years
    SAVE_PATH = "belgian_gp_analysis"
    VISUALIZATION_STYLE = "plotly"  # or "matplotlib"
    
    try:
        # ========================
        # 1. DATA LOADING
        # ========================
        print("Step 1: Loading historical Belgian GP data...")
        print("-" * 40)
        
        data_loader = BelgianGPDataLoader(years=YEARS)
        historical_data = data_loader.merge_all_data()
        
        if not historical_data:
            print("ERROR: No historical data could be loaded. Exiting.")
            return
        
        print(f"✓ Successfully loaded historical data:")
        for key, df in historical_data.items():
            if not df.empty:
                print(f"  - {key}: {len(df)} records")
        
        # Get current season data
        print("\n  Loading current season data...")
        current_season_data = data_loader.get_current_season_data()
        if not current_season_data.empty:
            print(f"✓ Current season data: {len(current_season_data)} records")
        else:
            print("⚠ No current season data available")
        
        # ========================
        # 2. FEATURE ENGINEERING
        # ========================
        print("\nStep 2: Feature Engineering...")
        print("-" * 40)
        
        feature_engineer = FeatureEngineer()
        
        # Process lap data if available
        if 'laps' in historical_data and not historical_data['laps'].empty:
            print("  Processing lap data...")
            featured_laps = feature_engineer.create_basic_features(historical_data['laps'])
            print(f"✓ Processed {len(featured_laps)} lap records")
        else:
            print("⚠ No lap data available for feature engineering")
            featured_laps = pd.DataFrame()
        
        # Create driver and team features
        if 'laps' in historical_data and 'results' in historical_data:
            print("  Creating driver and team features...")
            try:
                driver_features = feature_engineer.create_driver_features(
                    historical_data['laps'], historical_data['results']
                )
                team_features = feature_engineer.create_team_features(
                    historical_data['laps'], historical_data['results']
                )
                print(f"✓ Created features for {len(driver_features)} driver/year combinations")
            except Exception as e:
                print(f"⚠ Error creating driver/team features: {e}")
                driver_features = pd.DataFrame()
                team_features = pd.DataFrame()
        
        # Create tire features
        if 'laps' in historical_data and 'pit_stops' in historical_data:
            print("  Creating tire and pit stop features...")
            try:
                tire_features, pit_strategy, pit_windows = feature_engineer.create_tire_features(
                    historical_data['laps'], historical_data['pit_stops']
                )
                print(f"✓ Created tire features for {len(tire_features)} compound/year combinations")
            except Exception as e:
                print(f"⚠ Error creating tire features: {e}")
                tire_features = pd.DataFrame()
                pit_strategy = pd.DataFrame()
                pit_windows = pd.DataFrame()
        
        # ========================
        # 3. STRATEGY OPTIMIZATION
        # ========================
        print("\nStep 3: Pit Stop Strategy Optimization...")
        print("-" * 40)
        
        optimizer = PitStopOptimizer(race_length=44)  # Spa race length
        
        # Optimize strategies for teams
        optimization_results = {}
        
        if 'results' in historical_data and not historical_data['results'].empty:
            # Get unique teams from recent data
            recent_teams = historical_data['results'][
                historical_data['results']['Year'] >= 2022
            ]['Team'].unique() if 'Team' in historical_data['results'].columns else []
            
            print(f"  Optimizing strategies for {len(recent_teams)} teams...")
            
            for team in recent_teams[:5]:  # Limit to top 5 teams for demo
                try:
                    team_data = historical_data['results'][
                        historical_data['results']['Team'] == team
                    ]
                    
                    team_optimization = optimizer.optimize_for_team(
                        team_data, historical_data['results']
                    )
                    
                    if team_optimization:
                        optimization_results[team] = team_optimization
                        print(f"✓ Optimized strategy for {team}")
                except Exception as e:
                    print(f"⚠ Error optimizing strategy for {team}: {e}")
        
        # Generate strategy report
        if optimization_results:
            print("\n  Generating strategy reports...")
            for team, results in optimization_results.items():
                report = optimizer.generate_race_report(results)
                
                # Save report to file
                report_filename = f"{SAVE_PATH}_{team.replace(' ', '_')}_strategy_report.txt"
                with open(report_filename, 'w') as f:
                    f.write(report)
                print(f"✓ Strategy report saved: {report_filename}")
        
        # ========================
        # 4. PREDICTIVE MODELING
        # ========================
        print("\nStep 4: Race Winner Prediction...")
        print("-" * 40)
        
        predictor = BelgianGPPredictor()
        
        # Prepare training data
        if 'results' in historical_data and not historical_data['results'].empty:
            print("  Preparing training data...")
            X_train, y_train = predictor.prepare_training_data(
                historical_data, feature_engineer
            )
            
            if not X_train.empty and not y_train.empty:
                print(f"✓ Training data prepared: {len(X_train)} samples, {len(X_train.columns)} features")
                
                # Train models
                print("  Training prediction models...")
                training_results = predictor.train_models(X_train, y_train)
                
                if training_results:
                    print(f"✓ Models trained successfully. Best model: {predictor.best_model_name}")
                    
                    # Create prediction features for current drivers
                    if not current_season_data.empty:
                        print("  Creating prediction features...")
                        prediction_features = predictor.create_prediction_features(
                            current_season_data, historical_data
                        )
                        
                        if not prediction_features.empty:
                            print(f"✓ Prediction features created for {len(prediction_features)} drivers")
                            
                            # Make predictions
                            predictions = predictor.predict_winner_probabilities(prediction_features)
                            
                            if not predictions.empty:
                                print(f"✓ Predictions generated for {len(predictions)} drivers")
                                
                                # Display top predictions
                                print("\n  Top 5 Championship Contenders:")
                                for i, (_, row) in enumerate(predictions.head(5).iterrows()):
                                    print(f"    {i+1}. {row['Driver']}: {row['WinProbability']:.1%}")
                            else:
                                print("⚠ No predictions generated")
                        else:
                            print("⚠ No prediction features created")
                    else:
                        print("⚠ No current season data available for predictions")
                        predictions = pd.DataFrame()
                else:
                    print("⚠ Model training failed")
                    predictions = pd.DataFrame()
            else:
                print("⚠ No training data available")
                predictions = pd.DataFrame()
        else:
            print("⚠ No historical results data available")
            predictions = pd.DataFrame()
        
        # ========================
        # 5. VISUALIZATION
        # ========================
        print("\nStep 5: Generating Visualizations...")
        print("-" * 40)
        
        visualizer = F1Visualizer(style=VISUALIZATION_STYLE)
        
        # Create comprehensive dashboard
        try:
            # Individual plots
            print("  Creating tire degradation analysis...")
            if 'laps' in historical_data:
                visualizer.plot_tire_degradation(historical_data['laps'], SAVE_PATH)
            
            print("  Creating winning probability chart...")
            if not predictions.empty:
                visualizer.plot_winning_probabilities(predictions, SAVE_PATH)
            
            print("  Creating pit strategy analysis...")
            if optimization_results:
                # Convert team results to driver results for visualization
                all_driver_results = {}
                for team, team_results in optimization_results.items():
                    if 'drivers' in team_results:
                        all_driver_results.update(team_results['drivers'])
                
                if all_driver_results:
                    visualizer.plot_pit_strategy_comparison(
                        historical_data, {'drivers': all_driver_results}, SAVE_PATH
                    )
            
            print("  Creating stint length analysis...")
            if 'laps' in historical_data:
                visualizer.plot_stint_length_analysis(historical_data, SAVE_PATH)
            
            print("  Creating undercut/overcut analysis...")
            if 'pit_stops' in historical_data:
                visualizer.plot_undercut_overcut_analysis(historical_data, SAVE_PATH)
            
            # Feature importance plot
            if predictor.best_model is not None:
                print("  Creating feature importance plot...")
                model_summary = predictor.get_model_summary()
                visualizer.plot_feature_importance(model_summary, SAVE_PATH)
            
            print("✓ All visualizations generated successfully")
            
        except Exception as e:
            print(f"⚠ Error generating visualizations: {e}")
        
        # ========================
        # 6. FINAL REPORT
        # ========================
        print("\nStep 6: Generating Final Report...")
        print("-" * 40)
        
        report_lines = []
        report_lines.append("BELGIAN GRAND PRIX ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Data summary
        report_lines.append("DATA SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(f"Historical years analyzed: {min(YEARS)}-{max(YEARS)}")
        for key, df in historical_data.items():
            if not df.empty:
                report_lines.append(f"{key.title()}: {len(df)} records")
        report_lines.append("")
        
        # Prediction summary
        if not predictions.empty:
            report_lines.append("RACE WINNER PREDICTIONS")
            report_lines.append("-" * 30)
            report_lines.append(f"Model used: {predictor.best_model_name}")
            report_lines.append("Top 5 contenders:")
            for i, (_, row) in enumerate(predictions.head(5).iterrows()):
                report_lines.append(f"  {i+1}. {row['Driver']}: {row['WinProbability']:.1%}")
            report_lines.append("")
        
        # Strategy summary
        if optimization_results:
            report_lines.append("STRATEGY RECOMMENDATIONS")
            report_lines.append("-" * 30)
            for team, results in optimization_results.items():
                report_lines.append(f"{team}:")
                if 'team_strategy' in results:
                    approach = results['team_strategy'].get('recommended_approach', 'Unknown')
                    report_lines.append(f"  Recommended approach: {approach}")
                report_lines.append("")
        
        # Save final report
        final_report = "\n".join(report_lines)
        report_filename = f"{SAVE_PATH}_final_report.txt"
        with open(report_filename, 'w') as f:
            f.write(final_report)
        
        print(f"✓ Final report saved: {report_filename}")
        
        # ========================
        # COMPLETION
        # ========================
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved with prefix: {SAVE_PATH}")
        print()
        
        # Display key findings
        if not predictions.empty:
            top_contender = predictions.iloc[0]
            print(f"🏆 Top championship contender: {top_contender['Driver']} ({top_contender['WinProbability']:.1%})")
        
        if optimization_results:
            print(f"📊 Strategy recommendations generated for {len(optimization_results)} teams")
        
        print(f"📈 Visualizations saved as {VISUALIZATION_STYLE} files")
        print()
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

