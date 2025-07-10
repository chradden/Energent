#!/usr/bin/env python3
"""
ENERGENT Quick Start Script
Demonstrates basic functionality of the ENERGENT system
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_demo():
    """Run a quick demonstration of the ENERGENT system"""
    
    print("⚡ ENERGENT Quick Start Demo")
    print("=" * 50)
    
    try:
        # Import agents
        from agents.agent_a_heat_forecast import AgentAHeatForecast
        from agents.agent_b_price_forecast import AgentBPriceForecast
        
        print("\n1️⃣ Initializing Agents...")
        
        # Initialize Agent A (Heat Forecast)
        agent_a = AgentAHeatForecast(model_type="lstm")
        print("   ✅ Agent A (Heat Forecast) ready")
        
        # Initialize Agent B (Price Forecast)
        agent_b = AgentBPriceForecast(model_type="transformer")
        print("   ✅ Agent B (Price Forecast) ready")
        
        print("\n2️⃣ Generating Forecasts...")
        
        # Get weather data and generate heat forecast
        print("   \U0001F504 Generating heat demand forecast...")
        gas_csv_path = 'data/historical_Data/Gas usage combined_2024-01-01s.csv'
        agent_a.train_from_csv(gas_csv_path)
        heat_forecast_df = agent_a.predict_next_7_days()
        heat_forecast = heat_forecast_df['heat_demand_forecast'].tolist()
        heat_timestamps = heat_forecast_df['timestamp'].tolist()
        
        # Get price data and generate price forecast
        print("   🔄 Generating electricity price forecast...")
        price_data = agent_b.get_entsoe_data()
        agent_b.train(price_data)
        price_forecast = agent_b.predict(price_data)
        
        print("\n3️⃣ Running Optimization...")
        
        # Simple optimization (using the existing LP model)
        import pulp
        
        # Set up parameters
        global P_max_e, Q_max_th, Boiler_max, S_max, fuel_price
        P_max_e = 200.0
        Q_max_th = 300.0
        Boiler_max = 400.0
        S_max = 2000.0
        fuel_price = 0.03
        
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
            model += Q_CHP[t] + Q_boiler[t] + Q_discharge[t] == heat_forecast[t] + Q_charge[t]
            # Storage balance
            if t == 0:
                model += S[0] == S_max / 2
            else:
                model += S[t] == S[t-1] + Q_charge[t-1] - Q_discharge[t-1]
        
        # Objective function
        total_profit = pulp.lpSum([
            price_forecast[t] * P_CHP[t] 
            - fuel_price * (P_CHP[t] + Q_CHP[t]) / eta_total_CHP
            - fuel_price * Q_boiler[t] / eta_boiler
            for t in hours
        ])
        model += total_profit
        
        # Solve
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        print("   ✅ Optimization completed")
        
        print("\n4️⃣ Results Summary...")
        
        # Calculate key metrics
        total_revenue = sum(price_forecast[t] * P_CHP[t].value() for t in hours)
        total_cost = sum(fuel_price * (P_CHP[t].value() + Q_CHP[t].value()) / eta_total_CHP for t in hours)
        total_cost += sum(fuel_price * Q_boiler[t].value() / eta_boiler for t in hours)
        net_profit = total_revenue - total_cost
        
        print(f"   📊 Heat Demand (avg): {np.mean(heat_forecast):.1f} kW")
        print(f"   📊 Electricity Price (avg): {np.mean(price_forecast):.3f} €/kWh")
        print(f"   💰 Total Revenue: €{total_revenue:.2f}")
        print(f"   💰 Total Cost: €{total_cost:.2f}")
        print(f"   💰 Net Profit: €{net_profit:.2f}")
        print(f"   ⚡ CHP Power (avg): {np.mean([P_CHP[t].value() for t in hours]):.1f} kW")
        print(f"   🔥 CHP Heat (avg): {np.mean([Q_CHP[t].value() for t in hours]):.1f} kW")
        print(f"   🔥 Boiler Heat (avg): {np.mean([Q_boiler[t].value() for t in hours]):.1f} kW")
        
        print("\n5️⃣ Saving Results...")
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save forecasts
        forecast_df = pd.DataFrame({
            'timestamp': heat_timestamps,
            'heat_demand': heat_forecast,
            'electricity_price': price_forecast[:len(heat_forecast)]
        })
        forecast_df.to_csv('data/quick_demo_forecasts.csv', index=False)
        
        # Save optimization results
        results_df = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now(), periods=24, freq='H'),
            'chp_power': [P_CHP[t].value() for t in hours],
            'chp_heat': [Q_CHP[t].value() for t in hours],
            'boiler_heat': [Q_boiler[t].value() for t in hours],
            'storage': [S[t].value() for t in hours]
        })
        results_df.to_csv('data/quick_demo_results.csv', index=False)
        
        print("   ✅ Results saved to data/quick_demo_*.csv")
        
        print("\n🎉 Quick Demo Completed Successfully!")
        print("\n📁 Generated Files:")
        print("   - data/quick_demo_forecasts.csv")
        print("   - data/quick_demo_results.csv")
        
        print("\n🚀 Next Steps:")
        print("   - Run 'python main.py' for full pipeline")
        print("   - Run 'python main.py --dashboard' for web interface")
        print("   - Check the README.md for detailed documentation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check that you're in the correct directory")
        print("   3. Verify Python version (3.8+)")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'pandas', 'numpy', 'pulp', 'requests', 
        'sklearn', 'xgboost', 'prophet', 'yfinance'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies available!")
        return True

def main():
    """Main function"""
    print("⚡ ENERGENT Quick Start")
    print("=" * 30)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        return
    
    # Run demo
    success = quick_demo()
    
    if success:
        print("\n🎯 Demo completed! The ENERGENT system is ready to use.")
    else:
        print("\n💡 Try running the full system with: python main.py")

if __name__ == "__main__":
    main() 