import pulp
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class ExtendedCHPOptimizer:
    """Extended CHP optimizer with battery, PV, heat pump, and electric heaters"""
    
    def __init__(self, parameters_file: str = "data/parameters.json"):
        """Initialize optimizer with parameters"""
        self.load_parameters(parameters_file)
    
    def load_parameters(self, parameters_file: str):
        """Load system parameters"""
        try:
            with open(parameters_file, 'r') as f:
                self.params = json.load(f)['system_parameters']
            print("✅ Optimizer parameters loaded successfully")
        except Exception as e:
            print(f"❌ Error loading parameters: {e}")
            self.params = self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict:
        """Get default parameters"""
        return {
            "chp": {"P_max": 200.0, "Q_max": 300.0, "eta_el": 0.35, "eta_th": 0.50},
            "boiler": {"Q_max": 400.0, "eta": 0.90},
            "thermal_storage": {"capacity": 2000.0, "initial_level": 1000.0},
            "battery_storage": {"capacity": 100.0, "initial_level": 50.0},
            "photovoltaic": {"peak_power": 150.0, "efficiency": 0.18},
            "heat_pump": {"Q_max": 250.0, "P_max": 60.0, "cop_nominal": 4.2},
            "electric_heaters": {"Q_max": 200.0, "eta": 0.99},
            "economics": {"fuel_price": 0.03, "electricity_price_buy": 0.25}
        }
    
    def optimize_24h_schedule(self, heat_demand: List[float], 
                            electricity_prices: List[float],
                            pv_generation: List[float],
                            weather_data: List[Dict]) -> Dict:
        """Optimize 24-hour operation schedule"""
        
        print("🔧 Starting extended CHP optimization...")
        
        # Create optimization problem
        prob = pulp.LpProblem("Extended_CHP_Optimization", pulp.LpMaximize)
        
        # Time periods
        T = 24
        periods = range(T)
        
        # Decision variables
        # CHP operation
        chp_power = pulp.LpVariable.dicts("chp_power", periods, 0, self.params["chp"]["P_max"])
        chp_heat = pulp.LpVariable.dicts("chp_heat", periods, 0, self.params["chp"]["Q_max"])
        
        # Boiler operation
        boiler_heat = pulp.LpVariable.dicts("boiler_heat", periods, 0, self.params["boiler"]["Q_max"])
        
        # Heat pump operation
        heat_pump_heat = pulp.LpVariable.dicts("heat_pump_heat", periods, 0, self.params["heat_pump"]["Q_max"])
        heat_pump_power = pulp.LpVariable.dicts("heat_pump_power", periods, 0, self.params["heat_pump"]["P_max"])
        
        # Electric heaters
        electric_heater_heat = pulp.LpVariable.dicts("electric_heater_heat", periods, 0, self.params["electric_heaters"]["Q_max"])
        electric_heater_power = pulp.LpVariable.dicts("electric_heater_power", periods, 0, self.params["electric_heaters"]["Q_max"])
        
        # Battery storage
        battery_charge = pulp.LpVariable.dicts("battery_charge", periods, 0, self.params["battery_storage"]["max_charge_rate"])
        battery_discharge = pulp.LpVariable.dicts("battery_discharge", periods, 0, self.params["battery_storage"]["max_discharge_rate"])
        battery_level = pulp.LpVariable.dicts("battery_level", periods, 
                                             self.params["battery_storage"]["min_level"], 
                                             self.params["battery_storage"]["max_level"])
        
        # Thermal storage
        thermal_charge = pulp.LpVariable.dicts("thermal_charge", periods, 0, None)
        thermal_discharge = pulp.LpVariable.dicts("thermal_discharge", periods, 0, None)
        thermal_level = pulp.LpVariable.dicts("thermal_level", periods, 
                                             self.params["thermal_storage"]["min_level"], 
                                             self.params["thermal_storage"]["max_level"])
        
        # Grid exchange
        grid_import = pulp.LpVariable.dicts("grid_import", periods, 0, None)
        grid_export = pulp.LpVariable.dicts("grid_export", periods, 0, None)
        
        # Binary variables for component on/off states
        chp_on = pulp.LpVariable.dicts("chp_on", periods, 0, 1, cat='Binary')
        boiler_on = pulp.LpVariable.dicts("boiler_on", periods, 0, 1, cat='Binary')
        heat_pump_on = pulp.LpVariable.dicts("heat_pump_on", periods, 0, 1, cat='Binary')
        
        # Objective function: Maximize profit
        revenue = pulp.lpSum([
            chp_power[t] * electricity_prices[t] +  # CHP electricity revenue
            grid_export[t] * self.params["economics"]["electricity_price_sell"]  # Grid export revenue
            for t in periods
        ])
        
        costs = pulp.lpSum([
            # Fuel costs
            (chp_power[t] + chp_heat[t]) / (self.params["chp"]["eta_el"] + self.params["chp"]["eta_th"]) * self.params["economics"]["fuel_price"] +
            boiler_heat[t] / self.params["boiler"]["eta"] * self.params["economics"]["fuel_price"] +
            # Electricity costs
            grid_import[t] * electricity_prices[t] +
            # Maintenance costs (simplified)
            chp_on[t] * self.params["economics"]["maintenance_cost_chp"] +
            boiler_on[t] * self.params["economics"]["maintenance_cost_boiler"] +
            heat_pump_on[t] * self.params["economics"]["maintenance_cost_heat_pump"]
            for t in periods
        ])
        
        prob += revenue - costs
        
        # Constraints
        
        # 1. CHP heat-power relationship
        for t in periods:
            prob += chp_heat[t] == chp_power[t] * (self.params["chp"]["eta_th"] / self.params["chp"]["eta_el"])
            prob += chp_power[t] <= chp_on[t] * self.params["chp"]["P_max"]
            prob += chp_heat[t] <= chp_on[t] * self.params["chp"]["Q_max"]
        
        # 2. Heat pump COP relationship
        for t in periods:
            outdoor_temp = weather_data[t].get('temperature', 10)
            cop = self._calculate_cop(outdoor_temp)
            prob += heat_pump_power[t] == heat_pump_heat[t] / cop
            prob += heat_pump_heat[t] <= heat_pump_on[t] * self.params["heat_pump"]["Q_max"]
            prob += heat_pump_power[t] <= heat_pump_on[t] * self.params["heat_pump"]["P_max"]
        
        # 3. Electric heater efficiency
        for t in periods:
            prob += electric_heater_power[t] == electric_heater_heat[t] / self.params["electric_heaters"]["eta"]
        
        # 4. Heat balance
        for t in periods:
            prob += (chp_heat[t] + boiler_heat[t] + heat_pump_heat[t] + 
                    electric_heater_heat[t] + thermal_discharge[t] - thermal_charge[t] == heat_demand[t])
        
        # 5. Battery storage balance
        prob += battery_level[0] == self.params["battery_storage"]["initial_level"] + \
                battery_charge[0] * self.params["battery_storage"]["charge_efficiency"] - \
                battery_discharge[0] / self.params["battery_storage"]["discharge_efficiency"]
        
        for t in range(1, T):
            prob += battery_level[t] == battery_level[t-1] * (1 - self.params["battery_storage"]["self_discharge_rate"]) + \
                    battery_charge[t] * self.params["battery_storage"]["charge_efficiency"] - \
                    battery_discharge[t] / self.params["battery_storage"]["discharge_efficiency"]
        
        # 6. Thermal storage balance
        prob += thermal_level[0] == self.params["thermal_storage"]["initial_level"] + \
                thermal_charge[0] * self.params["thermal_storage"]["charge_efficiency"] - \
                thermal_discharge[0] / self.params["thermal_storage"]["discharge_efficiency"]
        
        for t in range(1, T):
            prob += thermal_level[t] == thermal_level[t-1] + \
                    thermal_charge[t] * self.params["thermal_storage"]["charge_efficiency"] - \
                    thermal_discharge[t] / self.params["thermal_storage"]["discharge_efficiency"]
        
        # 7. Electricity balance
        for t in periods:
            prob += (chp_power[t] + heat_pump_power[t] + electric_heater_power[t] + 
                    battery_charge[t] + grid_import[t] == 
                    pv_generation[t] + battery_discharge[t] + grid_export[t])
        
        # 8. Grid connection limit
        for t in periods:
            prob += grid_import[t] <= self.params.get("grid_connection_limit", 1000)
            prob += grid_export[t] <= self.params.get("grid_connection_limit", 1000)
        
        # 9. Component startup/shutdown constraints (simplified)
        for t in range(1, T):
            # Minimum uptime for CHP (2 hours)
            if t < T-1:
                prob += chp_on[t] >= chp_on[t-1] - chp_on[t+1]
        
        # Solve the problem
        print("🔍 Solving optimization problem...")
        prob.solve()
        
        if prob.status == pulp.LpStatusOptimal:
            print("✅ Optimization completed successfully!")
            
            # Extract results
            results = {
                'chp_power': [chp_power[t].value() for t in periods],
                'chp_heat': [chp_heat[t].value() for t in periods],
                'boiler_heat': [boiler_heat[t].value() for t in periods],
                'heat_pump_heat': [heat_pump_heat[t].value() for t in periods],
                'heat_pump_power': [heat_pump_power[t].value() for t in periods],
                'electric_heater_heat': [electric_heater_heat[t].value() for t in periods],
                'electric_heater_power': [electric_heater_power[t].value() for t in periods],
                'battery_charge': [battery_charge[t].value() for t in periods],
                'battery_discharge': [battery_discharge[t].value() for t in periods],
                'battery_level': [battery_level[t].value() for t in periods],
                'thermal_charge': [thermal_charge[t].value() for t in periods],
                'thermal_discharge': [thermal_discharge[t].value() for t in periods],
                'thermal_level': [thermal_level[t].value() for t in periods],
                'grid_import': [grid_import[t].value() for t in periods],
                'grid_export': [grid_export[t].value() for t in periods],
                'chp_on': [chp_on[t].value() for t in periods],
                'boiler_on': [boiler_on[t].value() for t in periods],
                'heat_pump_on': [heat_pump_on[t].value() for t in periods],
                'objective_value': pulp.value(prob.objective),
                'status': 'optimal'
            }
            
            # Calculate summary statistics
            total_revenue = sum(chp_power[t].value() * electricity_prices[t] + 
                              grid_export[t].value() * self.params["economics"]["electricity_price_sell"] 
                              for t in periods)
            total_cost = sum((chp_power[t].value() + chp_heat[t].value()) / 
                           (self.params["chp"]["eta_el"] + self.params["chp"]["eta_th"]) * 
                           self.params["economics"]["fuel_price"] +
                           boiler_heat[t].value() / self.params["boiler"]["eta"] * 
                           self.params["economics"]["fuel_price"] +
                           grid_import[t].value() * electricity_prices[t] 
                           for t in periods)
            
            results['summary'] = {
                'total_revenue': total_revenue,
                'total_cost': total_cost,
                'total_profit': total_revenue - total_cost,
                'total_heat_supplied': sum(results['chp_heat']) + sum(results['boiler_heat']) + 
                                     sum(results['heat_pump_heat']) + sum(results['electric_heater_heat']),
                'total_pv_used': sum(pv_generation),
                'total_grid_import': sum(results['grid_import']),
                'total_grid_export': sum(results['grid_export'])
            }
            
            return results
            
        else:
            print(f"❌ Optimization failed with status: {prob.status}")
            return {'status': 'failed', 'error': prob.status}
    
    def _calculate_cop(self, outdoor_temp: float) -> float:
        """Calculate heat pump COP based on outdoor temperature"""
        hp_params = self.params["heat_pump"]
        cop_nominal = hp_params["cop_nominal"]
        cop_min = hp_params["cop_min"]
        cop_max = hp_params["cop_max"]
        
        # Simplified COP calculation
        water_temp = 50  # Assumed water temperature
        temp_diff = water_temp - outdoor_temp
        cop_factor = 1 - 0.02 * temp_diff
        
        cop = cop_nominal * cop_factor
        return max(cop_min, min(cop_max, cop))
    
    def optimize_from_forecasts(self, heat_demand_forecast: pd.DataFrame,
                              electricity_price_forecast: pd.DataFrame,
                              pv_generation_forecast: pd.DataFrame,
                              weather_forecast: pd.DataFrame) -> Dict:
        """Optimize using forecast data"""
        
        # Extract 24-hour data
        heat_demand = heat_demand_forecast['heat_demand'].head(24).tolist()
        electricity_prices = electricity_price_forecast['electricity_price'].head(24).tolist()
        pv_generation = pv_generation_forecast['pv_generation'].head(24).tolist()
        
        # Convert weather data to list of dictionaries
        weather_data = []
        for _, row in weather_forecast.head(24).iterrows():
            weather_data.append({
                'temperature': row.get('temperature', 10),
                'solar_irradiance': row.get('solar_irradiance', 0),
                'humidity': row.get('humidity', 50)
            })
        
        return self.optimize_24h_schedule(heat_demand, electricity_prices, pv_generation, weather_data)

# Legacy function for backward compatibility
def optimize_chp_schedule(heat_demand: List[float], electricity_prices: List[float]) -> Dict:
    """Legacy optimization function"""
    optimizer = ExtendedCHPOptimizer()
    pv_generation = [0] * 24  # No PV in legacy mode
    weather_data = [{'temperature': 10}] * 24  # Default weather
    
    return optimizer.optimize_24h_schedule(heat_demand, electricity_prices, pv_generation, weather_data)

# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = ExtendedCHPOptimizer()
    
    # Sample data
    heat_demand = [200 + 50 * np.sin(2 * np.pi * t / 24) for t in range(24)]
    electricity_prices = [0.25 + 0.1 * np.sin(2 * np.pi * t / 24) for t in range(24)]
    pv_generation = [max(0, 100 * np.sin(2 * np.pi * (t - 6) / 24)) for t in range(24)]
    weather_data = [{'temperature': 15 + 10 * np.sin(2 * np.pi * t / 24)} for t in range(24)]
    
    # Run optimization
    results = optimizer.optimize_24h_schedule(heat_demand, electricity_prices, pv_generation, weather_data)
    
    if results['status'] == 'optimal':
        print(f"\n💰 Optimization Results:")
        print(f"Total profit: €{results['summary']['total_profit']:.2f}")
        print(f"Total revenue: €{results['summary']['total_revenue']:.2f}")
        print(f"Total cost: €{results['summary']['total_cost']:.2f}")
        print(f"Total heat supplied: {results['summary']['total_heat_supplied']:.1f} kWh")
        print(f"Total PV used: {results['summary']['total_pv_used']:.1f} kWh")
        
        # Save results
        results_df = pd.DataFrame({
            'hour': range(24),
            'chp_power': results['chp_power'],
            'chp_heat': results['chp_heat'],
            'boiler_heat': results['boiler_heat'],
            'heat_pump_heat': results['heat_pump_heat'],
            'electric_heater_heat': results['electric_heater_heat'],
            'battery_level': results['battery_level'],
            'thermal_level': results['thermal_level'],
            'grid_import': results['grid_import'],
            'grid_export': results['grid_export']
        })
        
        results_df.to_csv('data/optimization_results.csv', index=False)
        print("✅ Results saved to data/optimization_results.csv")
    else:
        print("❌ Optimization failed") 