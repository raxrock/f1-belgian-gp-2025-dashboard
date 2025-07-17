import numpy as np
import pandas as pd
from scipy.optimize import minimize
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class TireCompound:
    """Tire compound characteristics."""
    name: str
    base_pace: float  # seconds per lap advantage/disadvantage
    degradation_rate: float  # seconds per lap per lap
    pit_delta: float  # time lost during pit stop
    optimal_stint_length: Tuple[int, int]  # min, max optimal stint lengths

@dataclass
class Strategy:
    """Racing strategy definition."""
    pit_laps: List[int]
    compounds: List[str]
    total_time: float
    risk_factor: float

class PitStopOptimizer:
    """
    Pit stop strategy optimizer using Monte Carlo simulation and optimization.
    """
    
    def __init__(self, race_length=44):
        """
        Initialize the optimizer.
        
        Args:
            race_length (int): Number of laps in the race
        """
        self.race_length = race_length
        self.tire_compounds = {
            'SOFT': TireCompound('SOFT', -0.5, 0.08, 25.0, (8, 18)),
            'MEDIUM': TireCompound('MEDIUM', 0.0, 0.05, 25.0, (15, 30)),
            'HARD': TireCompound('HARD', 0.3, 0.03, 25.0, (20, 40))
        }
        
        # Spa-specific parameters
        self.pit_loss_time = 25.0  # seconds lost per pit stop
        self.undercut_advantage = 0.3  # seconds per lap advantage for fresh tires
        self.drs_zones = 3  # Number of DRS zones at Spa
        
    def calculate_tire_performance(self, compound: str, stint_length: int, lap_in_stint: int) -> float:
        """
        Calculate tire performance at a specific point in the stint.
        
        Args:
            compound (str): Tire compound name
            stint_length (int): Length of the stint
            lap_in_stint (int): Current lap in the stint
            
        Returns:
            float: Lap time penalty in seconds
        """
        tire = self.tire_compounds[compound]
        
        # Base pace difference
        base_time = tire.base_pace
        
        # Degradation component
        degradation_time = tire.degradation_rate * lap_in_stint
        
        # Thermal degradation (tires get worse as they heat up)
        thermal_degradation = 0.01 * (lap_in_stint ** 1.5)
        
        # Stint length penalty (very long stints are worse)
        if stint_length > tire.optimal_stint_length[1]:
            length_penalty = 0.05 * (stint_length - tire.optimal_stint_length[1])
        else:
            length_penalty = 0
        
        return base_time + degradation_time + thermal_degradation + length_penalty
    
    def simulate_race_time(self, strategy: List[Tuple[int, str]], driver_skill: float = 1.0) -> float:
        """
        Simulate total race time for a given strategy.
        
        Args:
            strategy (List[Tuple[int, str]]): List of (stint_length, compound) tuples
            driver_skill (float): Driver skill factor (0.8-1.2)
            
        Returns:
            float: Total race time in seconds
        """
        total_time = 0
        current_lap = 0
        stint_number = 0
        
        for stint_length, compound in strategy:
            # Add pit stop time (except first stint)
            if stint_number > 0:
                total_time += self.pit_loss_time
            
            # Calculate stint time
            for lap_in_stint in range(1, stint_length + 1):
                current_lap += 1
                if current_lap > self.race_length:
                    break
                
                # Base lap time (hypothetical perfect lap)
                base_lap_time = 104.0  # Spa average lap time in seconds
                
                # Tire performance
                tire_time = self.calculate_tire_performance(compound, stint_length, lap_in_stint)
                
                # Driver skill factor
                skill_factor = (2 - driver_skill) * 0.2  # Convert to time penalty
                
                # Random variation (track conditions, traffic, etc.)
                random_factor = np.random.normal(0, 0.5)
                
                lap_time = base_lap_time + tire_time + skill_factor + random_factor
                total_time += lap_time
            
            stint_number += 1
            
            if current_lap >= self.race_length:
                break
        
        return total_time
    
    def calculate_undercut_potential(self, current_position: int, gap_to_ahead: float, pit_lap: int) -> float:
        """
        Calculate undercut potential for a given pit stop timing.
        
        Args:
            current_position (int): Current track position
            gap_to_ahead (float): Gap to car ahead in seconds
            pit_lap (int): Proposed pit stop lap
            
        Returns:
            float: Undercut advantage in seconds
        """
        # Fresh tire advantage
        fresh_tire_advantage = self.undercut_advantage
        
        # Track position factor (harder to undercut from further back)
        position_factor = max(0.5, 1.0 - (current_position - 1) * 0.05)
        
        # Gap factor (easier to undercut with smaller gaps)
        gap_factor = max(0.2, 1.0 - gap_to_ahead / 30.0)
        
        # Pit window factor (optimal pit windows are more effective)
        optimal_pit_window = range(15, 25)
        if pit_lap in optimal_pit_window:
            window_factor = 1.0
        else:
            window_factor = 0.7
        
        return fresh_tire_advantage * position_factor * gap_factor * window_factor
    
    def generate_strategy_variants(self, base_strategy: str) -> List[List[Tuple[int, str]]]:
        """
        Generate strategy variants for Monte Carlo simulation.
        
        Args:
            base_strategy (str): Base strategy type ('1-stop', '2-stop', '3-stop')
            
        Returns:
            List[List[Tuple[int, str]]]: List of strategy variants
        """
        variants = []
        
        if base_strategy == '1-stop':
            # One pit stop strategies
            for pit_lap in range(18, 28):
                stint1_length = pit_lap
                stint2_length = self.race_length - pit_lap
                
                # Different compound combinations
                compound_combos = [
                    ('MEDIUM', 'HARD'),
                    ('SOFT', 'MEDIUM'),
                    ('HARD', 'MEDIUM')
                ]
                
                for c1, c2 in compound_combos:
                    if stint1_length <= self.tire_compounds[c1].optimal_stint_length[1] and \
                       stint2_length <= self.tire_compounds[c2].optimal_stint_length[1]:
                        variants.append([(stint1_length, c1), (stint2_length, c2)])
        
        elif base_strategy == '2-stop':
            # Two pit stop strategies
            for pit1_lap in range(12, 22):
                for pit2_lap in range(pit1_lap + 10, 35):
                    stint1_length = pit1_lap
                    stint2_length = pit2_lap - pit1_lap
                    stint3_length = self.race_length - pit2_lap
                    
                    # Different compound combinations
                    compound_combos = [
                        ('MEDIUM', 'HARD', 'SOFT'),
                        ('SOFT', 'MEDIUM', 'HARD'),
                        ('HARD', 'MEDIUM', 'SOFT'),
                        ('MEDIUM', 'MEDIUM', 'SOFT')
                    ]
                    
                    for c1, c2, c3 in compound_combos:
                        if (stint1_length <= self.tire_compounds[c1].optimal_stint_length[1] and
                            stint2_length <= self.tire_compounds[c2].optimal_stint_length[1] and
                            stint3_length <= self.tire_compounds[c3].optimal_stint_length[1]):
                            variants.append([(stint1_length, c1), (stint2_length, c2), (stint3_length, c3)])
        
        elif base_strategy == '3-stop':
            # Three pit stop strategies (aggressive)
            for pit1_lap in range(10, 16):
                for pit2_lap in range(pit1_lap + 8, pit1_lap + 14):
                    for pit3_lap in range(pit2_lap + 8, pit2_lap + 14):
                        if pit3_lap >= self.race_length - 8:
                            continue
                        
                        stint1_length = pit1_lap
                        stint2_length = pit2_lap - pit1_lap
                        stint3_length = pit3_lap - pit2_lap
                        stint4_length = self.race_length - pit3_lap
                        
                        # High-performance compound combinations
                        compound_combos = [
                            ('SOFT', 'SOFT', 'MEDIUM', 'HARD'),
                            ('MEDIUM', 'SOFT', 'SOFT', 'MEDIUM')
                        ]
                        
                        for c1, c2, c3, c4 in compound_combos:
                            variants.append([(stint1_length, c1), (stint2_length, c2), 
                                           (stint3_length, c3), (stint4_length, c4)])
        
        return variants
    
    def monte_carlo_simulation(self, strategy_variants: List[List[Tuple[int, str]]], 
                               driver_skill: float = 1.0, n_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation for strategy variants.
        
        Args:
            strategy_variants (List): List of strategy variants
            driver_skill (float): Driver skill factor
            n_simulations (int): Number of simulations per variant
            
        Returns:
            Dict: Simulation results
        """
        results = {}
        
        for i, strategy in enumerate(strategy_variants):
            times = []
            for _ in range(n_simulations):
                race_time = self.simulate_race_time(strategy, driver_skill)
                times.append(race_time)
            
            results[f'strategy_{i}'] = {
                'strategy': strategy,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'percentile_95': np.percentile(times, 95),
                'percentile_5': np.percentile(times, 5),
                'risk_factor': np.std(times) / np.mean(times)
            }
        
        return results
    
    def optimize_for_driver(self, driver_data: pd.DataFrame, historical_data: pd.DataFrame,
                           current_position: int = 10, gap_to_ahead: float = 5.0) -> Dict:
        """
        Optimize strategy for a specific driver.
        
        Args:
            driver_data (pd.DataFrame): Driver's historical data
            historical_data (pd.DataFrame): Historical race data
            current_position (int): Current championship position
            gap_to_ahead (float): Gap to car ahead in seconds
            
        Returns:
            Dict: Optimized strategy recommendations
        """
        # Calculate driver skill factor based on historical performance
        if not driver_data.empty:
            avg_position = driver_data['Position'].mean() if 'Position' in driver_data.columns else 10
            driver_skill = max(0.8, min(1.2, 1.3 - (avg_position - 1) * 0.05))
        else:
            driver_skill = 1.0
        
        # Generate strategy variants
        strategy_types = ['1-stop', '2-stop']
        if current_position > 5:  # More aggressive for lower positions
            strategy_types.append('3-stop')
        
        all_variants = []
        for strategy_type in strategy_types:
            variants = self.generate_strategy_variants(strategy_type)
            all_variants.extend(variants)
        
        # Run simulations
        simulation_results = self.monte_carlo_simulation(all_variants, driver_skill)
        
        # Find optimal strategies
        best_overall = min(simulation_results.items(), key=lambda x: x[1]['mean_time'])
        best_conservative = min(simulation_results.items(), key=lambda x: x[1]['risk_factor'])
        best_aggressive = min(simulation_results.items(), key=lambda x: x[1]['percentile_5'])
        
        # Calculate undercut opportunities
        undercut_opportunities = []
        for pit_lap in range(12, 30):
            undercut_potential = self.calculate_undercut_potential(current_position, gap_to_ahead, pit_lap)
            undercut_opportunities.append({
                'lap': pit_lap,
                'potential': undercut_potential,
                'recommended': undercut_potential > 0.5
            })
        
        return {
            'driver_skill_factor': driver_skill,
            'recommended_strategy': best_overall[1]['strategy'],
            'expected_time': best_overall[1]['mean_time'],
            'alternative_strategies': {
                'conservative': best_conservative[1]['strategy'],
                'aggressive': best_aggressive[1]['strategy']
            },
            'undercut_opportunities': undercut_opportunities,
            'risk_analysis': {
                'recommended_risk': best_overall[1]['risk_factor'],
                'time_variance': best_overall[1]['std_time']
            }
        }
    
    def optimize_for_team(self, team_data: pd.DataFrame, historical_data: pd.DataFrame) -> Dict:
        """
        Optimize strategy for an entire team.
        
        Args:
            team_data (pd.DataFrame): Team's historical data
            historical_data (pd.DataFrame): Historical race data
            
        Returns:
            Dict: Team strategy recommendations
        """
        team_strategies = {}
        
        # Get unique drivers for the team
        drivers = team_data['Driver'].unique() if 'Driver' in team_data.columns else []
        
        for i, driver in enumerate(drivers):
            driver_data = team_data[team_data['Driver'] == driver]
            
            # Estimate current position based on historical performance
            current_position = int(driver_data['Position'].mean()) if 'Position' in driver_data.columns else 10
            
            # Optimize strategy for this driver
            driver_strategy = self.optimize_for_driver(
                driver_data, historical_data, current_position, 5.0
            )
            
            team_strategies[driver] = driver_strategy
        
        # Team-level strategic considerations
        team_recommendations = {
            'drivers': team_strategies,
            'team_strategy': self._analyze_team_strategy(team_strategies),
            'strategic_flexibility': self._calculate_strategic_flexibility(team_strategies)
        }
        
        return team_recommendations
    
    def _analyze_team_strategy(self, team_strategies: Dict) -> Dict:
        """
        Analyze team-level strategy patterns.
        
        Args:
            team_strategies (Dict): Individual driver strategies
            
        Returns:
            Dict: Team strategy analysis
        """
        if not team_strategies:
            return {}
        
        # Count strategy types
        strategy_counts = {'1-stop': 0, '2-stop': 0, '3-stop': 0}
        
        for driver, strategy in team_strategies.items():
            num_stops = len(strategy['recommended_strategy'])
            if num_stops == 2:
                strategy_counts['1-stop'] += 1
            elif num_stops == 3:
                strategy_counts['2-stop'] += 1
            elif num_stops == 4:
                strategy_counts['3-stop'] += 1
        
        # Recommend team approach
        if strategy_counts['1-stop'] > strategy_counts['2-stop']:
            team_approach = 'Conservative (1-stop focus)'
        elif strategy_counts['2-stop'] > 0:
            team_approach = 'Balanced (2-stop focus)'
        else:
            team_approach = 'Aggressive (3-stop focus)'
        
        return {
            'strategy_distribution': strategy_counts,
            'recommended_approach': team_approach,
            'strategic_diversity': len([k for k, v in strategy_counts.items() if v > 0])
        }
    
    def _calculate_strategic_flexibility(self, team_strategies: Dict) -> float:
        """
        Calculate team's strategic flexibility score.
        
        Args:
            team_strategies (Dict): Individual driver strategies
            
        Returns:
            float: Flexibility score (0-1)
        """
        if not team_strategies:
            return 0.0
        
        # Calculate based on risk variance and strategy diversity
        risk_scores = []
        for driver, strategy in team_strategies.items():
            risk_scores.append(strategy['risk_analysis']['recommended_risk'])
        
        risk_variance = np.var(risk_scores) if risk_scores else 0
        flexibility_score = min(1.0, risk_variance * 10)  # Normalize to 0-1
        
        return flexibility_score
    
    def generate_race_report(self, optimization_results: Dict) -> str:
        """
        Generate a comprehensive race strategy report.
        
        Args:
            optimization_results (Dict): Results from optimization
            
        Returns:
            str: Formatted race report
        """
        report = []
        report.append("=== BELGIAN GRAND PRIX STRATEGY REPORT ===\n")
        
        if 'drivers' in optimization_results:
            # Team report
            report.append("TEAM STRATEGY ANALYSIS")
            report.append("-" * 40)
            
            team_strategy = optimization_results['team_strategy']
            report.append(f"Recommended Approach: {team_strategy['recommended_approach']}")
            report.append(f"Strategic Diversity: {team_strategy['strategic_diversity']}/3")
            report.append(f"Strategic Flexibility: {optimization_results['strategic_flexibility']:.2f}")
            report.append("")
            
            # Individual driver recommendations
            for driver, strategy in optimization_results['drivers'].items():
                report.append(f"DRIVER: {driver}")
                report.append("-" * 20)
                
                recommended = strategy['recommended_strategy']
                report.append(f"Recommended Strategy: {len(recommended)}-stop")
                
                for i, (stint_length, compound) in enumerate(recommended):
                    report.append(f"  Stint {i+1}: {compound} ({stint_length} laps)")
                
                report.append(f"Expected Race Time: {strategy['expected_time']:.1f}s")
                report.append(f"Risk Factor: {strategy['risk_analysis']['recommended_risk']:.3f}")
                report.append("")
                
                # Undercut opportunities
                best_undercuts = sorted(strategy['undercut_opportunities'], 
                                      key=lambda x: x['potential'], reverse=True)[:3]
                report.append("Top Undercut Opportunities:")
                for opp in best_undercuts:
                    status = "RECOMMENDED" if opp['recommended'] else "POSSIBLE"
                    report.append(f"  Lap {opp['lap']}: {opp['potential']:.2f}s advantage ({status})")
                
                report.append("")
        
        return "\n".join(report)