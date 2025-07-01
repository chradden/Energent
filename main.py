#!/usr/bin/env python3
"""
ENERGENT - Main Orchestrator
Intelligent Multi-Agent System for CHP Optimization

This script orchestrates all agents and provides a command-line interface
for running the complete ENERGENT system.
"""

import argparse
import json
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.agent_a_heat_forecast import AgentAHeatForecast
from agents.agent_b_price_forecast import AgentBPriceForecast
from agents.agent_c_optimizer_lp import *
from agents.agent_c_optimizer_rl import *
from simulations.agent_d_simulation import *

class ENERGENTOrchestrator:
    """Main orchestrator for the ENERGENT system"""
    
    def __init__(self, config_file: str = "data/parameters.json"):
        """Initialize the orchestrator with configuration"""
        self.config = self.load_config(config_file)
        self.agents = {}
        self.results = {}
        
    def load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_file} not found. Using defaults.")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            "system_parameters": {
                "chp": {"P_max": 200.0, "Q_max": 300.0},
                "boiler": {"Q_max": 400.0},
                "storage": {"capacity": 2000.0}
            },
            "forecasting": {"horizon": 24},
            "optimization": {"default_method": "linear_programming"}
        }
    
    def initialize_agents(self, heat_model: str = "lstm", price_model: str = "transformer"):
        """Initialize all agents"""
        print("🤖 Initializing ENERGENT agents...")
        
        # Agent A - Heat Forecast
        self.agents['heat_forecast'] = AgentAHeatForecast(model_type=heat_model)
        print("✅ Agent A (Heat Forecast) initialized")
        
        # Agent B - Price Forecast
        self.agents['price_forecast'] = AgentBPriceForecast(model_type=price_model)
        print("✅ Agent B (Price Forecast) initialized")
        
        print("✅ All agents initialized successfully!")
    
    def run_forecasting_pipeline(self) -> dict:
        """Run the forecasting pipeline (Agents A & B)"""
        print("\n📈 Running forecasting pipeline...")
        
        # Agent A - Heat Demand Forecast
        print("🔄 Agent A: Generating heat demand forecast...")
        weather_data = self.agents['heat_forecast'].get_weather_data()
        self.agents['heat_forecast'].train(weather_data)
        heat_forecast = self.agents['heat_forecast'].predict(weather_data)
        
        # Agent B - Electricity Price Forecast
        print("🔄 Agent B: Generating electricity price forecast...")
        price_data = self.agents['price_forecast'].get_entsoe_data()
        self.agents['price_forecast'].train(price_data)
        price_forecast = self.agents['price_forecast'].predict(price_data)
        
        # Store forecasts
        forecasts = {
            'heat_demand': heat_forecast,
            'electricity_price': price_forecast,
            'timestamps': pd.date_range(start=datetime.now(), periods=24, freq='H')
        }
        
        # Save forecasts
        self.save_forecasts(forecasts)
        
        print("✅ Forecasting pipeline completed!")
        return forecasts
    
    def run_optimization_pipeline(self, forecasts: dict, method: str = "linear_programming") -> dict:
        """Run the optimization pipeline (Agent C)"""
        print(f"\n🎯 Running optimization pipeline ({method})...")
        
        # Extract system parameters
        chp_config = self.config['system_parameters']['chp']
        boiler_config = self.config['system_parameters']['boiler']
        storage_config = self.config['system_parameters']['storage']
        economics = self.config['system_parameters']['economics']
        
        if method == "linear_programming":
            results = self.run_lp_optimization(
                forecasts['heat_demand'],
                forecasts['electricity_price'],
                chp_config['P_max'],
                chp_config['Q_max'],
                boiler_config['Q_max'],
                storage_config['capacity'],
                economics['fuel_price']
            )
        elif method == "reinforcement_learning":
            results = self.run_rl_optimization(
                forecasts['heat_demand'],
                forecasts['electricity_price'],
                chp_config['P_max'],
                chp_config['Q_max'],
                boiler_config['Q_max'],
                storage_config['capacity'],
                economics['fuel_price']
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Save optimization results
        self.save_optimization_results(results)
        
        print("✅ Optimization pipeline completed!")
        return results
    
    def run_lp_optimization(self, heat_demand, elec_price, chp_power_max, chp_heat_max, 
                           boiler_max, storage_capacity, fuel_price):
        """Run Linear Programming optimization"""
        
        # Update global variables for the LP model
        global P_max_e, Q_max_th, Boiler_max, S_max, fuel_price_global
        P_max_e = chp_power_max
        Q_max_th = chp_heat_max
        Boiler_max = boiler_max
        S_max = storage_capacity
        fuel_price_global = fuel_price
        
        # Create LP model
        model = pulp.LpProblem("CHP_schedule", pulp.LpMaximize)
        
        # Variables
        hours = range(24)
        P_CHP = pulp.LpVariable.dicts("P_CHP_elec", hours, lowBound=0, upBound=P_max_e)
        Q_CHP = pulp.LpVariable.dicts("Q_CHP_heat", hours, lowBound=0, upBound=Q_max_th)
        Q_boiler = pulp.LpVariable.dicts("Q_boiler", hours, lowBound=0, upBound=Boiler_max)
        Q_charge = pulp.LpVariable.dicts("Q_charge", hours, lowBound=0)
        Q_discharge = pulp.LpVariable.dicts("Q_discharge", hours, lowBound=0)
        S = pulp.LpVariable.dicts("Storage", hours, lowBound=0, upBound=S_max)
        
        # Constraints
        alpha = Q_max_th / P_max_e
        eta_total_CHP = 0.85
        eta_boiler = 0.90
        
        for t in hours:
            # CHP coupling
            model += Q_CHP[t] == alpha * P_CHP[t]
            # Heat balance
            model += Q_CHP[t] + Q_boiler[t] + Q_discharge[t] == heat_demand[t] + Q_charge[t]
            # Storage balance
            if t == 0:
                model += S[0] == S_max / 2
            else:
                model += S[t] == S[t-1] + Q_charge[t-1] - Q_discharge[t-1]
        
        # Objective function
        total_profit = pulp.lpSum([
            elec_price[t] * P_CHP[t] 
            - fuel_price * (P_CHP[t] + Q_CHP[t]) / eta_total_CHP
            - fuel_price * Q_boiler[t] / eta_boiler
            for t in hours
        ])
        model += total_profit
        
        # Solve
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract results
        results = {
            'method': 'Linear Programming',
            'total_profit': pulp.value(total_profit),
            'chp_power': [P_CHP[t].value() for t in hours],
            'chp_heat': [Q_CHP[t].value() for t in hours],
            'boiler_heat': [Q_boiler[t].value() for t in hours],
            'storage': [S[t].value() for t in hours],
            'charge': [Q_charge[t].value() for t in hours],
            'discharge': [Q_discharge[t].value() for t in hours]
        }
        
        return results
    
    def run_rl_optimization(self, heat_demand, elec_price, chp_power_max, chp_heat_max, 
                           boiler_max, storage_capacity, fuel_price):
        """Run Reinforcement Learning optimization"""
        
        # Create environment
        env = CHPEnv(heat_demand, elec_price, initial_storage=storage_capacity/2)
        
        # Simple policy (for demo purposes)
        total_reward = 0
        actions = []
        
        state = env.reset()
        for t in range(24):
            # Simple heuristic policy
            demand_ratio = heat_demand[t] / (chp_heat_max + boiler_max)
            price_ratio = elec_price[t] / max(elec_price)
            
            # Action based on demand and price
            chp_action = min(demand_ratio * 0.8, 1.0)
            boiler_action = max(0, demand_ratio - chp_action)
            
            action = np.array([chp_action, boiler_action])
            actions.append(action)
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        # Extract results
        results = {
            'method': 'Reinforcement Learning',
            'total_profit': total_reward,
            'chp_power': [actions[t][0] * chp_power_max for t in range(24)],
            'chp_heat': [actions[t][0] * chp_heat_max for t in range(24)],
            'boiler_heat': [actions[t][1] * boiler_max for t in range(24)],
            'storage': [storage_capacity/2] * 24,  # Simplified
            'charge': [0] * 24,  # Simplified
            'discharge': [0] * 24  # Simplified
        }
        
        return results
    
    def run_simulation_pipeline(self, optimization_results: dict, forecasts: dict) -> dict:
        """Run the simulation pipeline (Agent D)"""
        print("\n🔄 Running simulation pipeline...")
        
        # Create simulation data
        simulation_data = {
            'optimization_results': optimization_results,
            'forecasts': forecasts,
            'config': self.config
        }
        
        # Run simulation (simplified version)
        simulation_results = self.run_simulation(simulation_data)
        
        # Save simulation results
        self.save_simulation_results(simulation_results)
        
        print("✅ Simulation pipeline completed!")
        return simulation_results
    
    def run_simulation(self, data: dict) -> dict:
        """Run simulation to evaluate optimization results"""
        
        opt_results = data['optimization_results']
        forecasts = data['forecasts']
        
        # Calculate additional metrics
        total_revenue = sum(opt_results['chp_power'][t] * forecasts['electricity_price'][t] for t in range(24))
        total_cost = sum(opt_results['chp_power'][t] * 0.03 for t in range(24))  # Simplified fuel cost
        net_profit = total_revenue - total_cost
        
        # Calculate efficiency
        total_heat_produced = sum(opt_results['chp_heat']) + sum(opt_results['boiler_heat'])
        total_heat_demand = sum(forecasts['heat_demand'])
        efficiency = (total_heat_demand / total_heat_produced) * 100 if total_heat_produced > 0 else 0
        
        # Calculate CO2 emissions (simplified)
        fuel_consumption = sum(opt_results['chp_power']) / 0.85 + sum(opt_results['boiler_heat']) / 0.90
        co2_emissions = fuel_consumption * 0.2  # kg CO2 per kWh fuel
        
        return {
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'net_profit': net_profit,
            'efficiency': efficiency,
            'co2_emissions': co2_emissions,
            'fuel_consumption': fuel_consumption,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_forecasts(self, forecasts: dict):
        """Save forecasts to files"""
        # Save heat demand forecast
        heat_df = pd.DataFrame({
            'timestamp': forecasts['timestamps'],
            'heat_demand_forecast': forecasts['heat_demand']
        })
        heat_df.to_csv('data/heat_demand_forecast.csv', index=False)
        
        # Save electricity price forecast
        price_df = pd.DataFrame({
            'timestamp': forecasts['timestamps'],
            'electricity_price_forecast': forecasts['electricity_price']
        })
        price_df.to_csv('data/electricity_price_forecast.csv', index=False)
    
    def save_optimization_results(self, results: dict):
        """Save optimization results to file"""
        # Create control schedule
        control_df = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now(), periods=24, freq='H'),
            'action_bhkw': [p / self.config['system_parameters']['chp']['P_max'] for p in results['chp_power']],
            'action_boiler': [q / self.config['system_parameters']['boiler']['Q_max'] for q in results['boiler_heat']]
        })
        control_df.to_csv('data/control_schedule.csv', index=False)
        
        # Save detailed results
        with open('data/optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def save_simulation_results(self, results: dict):
        """Save simulation results to file"""
        with open('data/simulation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def print_summary(self, forecasts: dict, optimization_results: dict, simulation_results: dict):
        """Print a summary of all results"""
        print("\n" + "="*60)
        print("📊 ENERGENT SYSTEM SUMMARY")
        print("="*60)
        
        print(f"\n🔮 FORECASTS (24-hour horizon):")
        print(f"   Heat Demand: {np.mean(forecasts['heat_demand']):.1f} kW (avg)")
        print(f"   Electricity Price: {np.mean(forecasts['electricity_price']):.3f} €/kWh (avg)")
        
        print(f"\n🎯 OPTIMIZATION RESULTS ({optimization_results['method']}):")
        print(f"   Total Profit: €{optimization_results['total_profit']:.2f}")
        print(f"   CHP Power: {np.mean(optimization_results['chp_power']):.1f} kW (avg)")
        print(f"   CHP Heat: {np.mean(optimization_results['chp_heat']):.1f} kW (avg)")
        print(f"   Boiler Heat: {np.mean(optimization_results['boiler_heat']):.1f} kW (avg)")
        
        print(f"\n📈 SIMULATION RESULTS:")
        print(f"   Total Revenue: €{simulation_results['total_revenue']:.2f}")
        print(f"   Total Cost: €{simulation_results['total_cost']:.2f}")
        print(f"   Net Profit: €{simulation_results['net_profit']:.2f}")
        print(f"   Efficiency: {simulation_results['efficiency']:.1f}%")
        print(f"   CO2 Emissions: {simulation_results['co2_emissions']:.1f} kg")
        
        print("\n" + "="*60)
    
    def run_complete_pipeline(self, heat_model: str = "lstm", price_model: str = "transformer", 
                            opt_method: str = "linear_programming"):
        """Run the complete ENERGENT pipeline"""
        print("🚀 Starting ENERGENT complete pipeline...")
        
        # Initialize agents
        self.initialize_agents(heat_model, price_model)
        
        # Run forecasting pipeline
        forecasts = self.run_forecasting_pipeline()
        
        # Run optimization pipeline
        optimization_results = self.run_optimization_pipeline(forecasts, opt_method)
        
        # Run simulation pipeline
        simulation_results = self.run_simulation_pipeline(optimization_results, forecasts)
        
        # Print summary
        self.print_summary(forecasts, optimization_results, simulation_results)
        
        print("✅ ENERGENT pipeline completed successfully!")
        return {
            'forecasts': forecasts,
            'optimization': optimization_results,
            'simulation': simulation_results
        }

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="ENERGENT - CHP Optimization System")
    parser.add_argument("--config", default="data/parameters.json", help="Configuration file path")
    parser.add_argument("--heat-model", default="lstm", choices=["lstm", "xgboost", "prophet"], 
                       help="Heat forecasting model")
    parser.add_argument("--price-model", default="transformer", choices=["transformer", "lstm"], 
                       help="Price forecasting model")
    parser.add_argument("--opt-method", default="linear_programming", 
                       choices=["linear_programming", "reinforcement_learning"], 
                       help="Optimization method")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = ENERGENTOrchestrator(args.config)
    
    if args.dashboard:
        print("🌐 Launching Streamlit dashboard...")
        os.system("streamlit run dashboard/app.py")
    else:
        # Run complete pipeline
        results = orchestrator.run_complete_pipeline(
            heat_model=args.heat_model,
            price_model=args.price_model,
            opt_method=args.opt_method
        )
        
        print(f"\n📁 Results saved to data/ directory")
        print("🌐 Run 'python main.py --dashboard' to launch the web interface")

if __name__ == "__main__":
    main() 