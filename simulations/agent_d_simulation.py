import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class ExtendedCHPSimulation:
    """Extended simulation environment for CHP system with battery, PV, heat pump, and electric heaters"""
    
    def __init__(self, parameters_file: str = "data/parameters.json"):
        """Initialize simulation with parameters"""
        self.load_parameters(parameters_file)
        self.reset_simulation()
    
    def load_parameters(self, parameters_file: str):
        """Load system parameters from JSON file"""
        try:
            with open(parameters_file, 'r') as f:
                self.params = json.load(f)['system_parameters']
            print("✅ System parameters loaded successfully")
        except Exception as e:
            print(f"❌ Error loading parameters: {e}")
            self.params = self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict:
        """Get default parameters if file loading fails"""
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
    
    def reset_simulation(self):
        """Reset simulation state"""
        self.thermal_storage_level = self.params["thermal_storage"]["initial_level"]
        self.battery_level = self.params["battery_storage"]["initial_level"]
        self.results = []
        self.timestamps = []
    
    def calculate_pv_generation(self, timestamp: datetime, weather_data: Dict) -> float:
        """Calculate PV generation based on weather and time"""
        # Extract weather data
        solar_irradiance = weather_data.get('solar_irradiance', 0)  # W/m²
        temperature = weather_data.get('temperature', 20)  # °C
        
        # PV system parameters
        pv_params = self.params["photovoltaic"]
        peak_power = pv_params["peak_power"]  # kW
        efficiency = pv_params["efficiency"]
        temp_coeff = pv_params["temperature_coefficient"]
        inverter_eff = pv_params["inverter_efficiency"]
        soiling = pv_params["soiling_factor"]
        shading = pv_params["shading_factor"]
        
        # Temperature correction
        temp_factor = 1 + temp_coeff * (temperature - 25)
        
        # Calculate generation
        generation = (solar_irradiance / 1000) * peak_power * efficiency * temp_factor * inverter_eff * soiling * shading
        
        return max(0, generation)
    
    def calculate_heat_pump_cop(self, outdoor_temp: float, water_temp: float) -> float:
        """Calculate COP of heat pump based on temperatures"""
        hp_params = self.params["heat_pump"]
        cop_nominal = hp_params["cop_nominal"]
        cop_min = hp_params["cop_min"]
        cop_max = hp_params["cop_max"]
        
        # Simplified COP calculation based on temperature difference
        temp_diff = water_temp - outdoor_temp
        cop_factor = 1 - 0.02 * temp_diff  # COP decreases with temperature difference
        
        cop = cop_nominal * cop_factor
        return max(cop_min, min(cop_max, cop))
    
    def simulate_timestep(self, timestamp: datetime, heat_demand: float, 
                         electricity_price: float, weather_data: Dict,
                         control_actions: Dict) -> Dict:
        """Simulate one timestep of the system"""
        
        # Extract control actions
        chp_power = control_actions.get('chp_power', 0)
        boiler_heat = control_actions.get('boiler_heat', 0)
        heat_pump_heat = control_actions.get('heat_pump_heat', 0)
        electric_heater_heat = control_actions.get('electric_heater_heat', 0)
        battery_charge = control_actions.get('battery_charge', 0)
        battery_discharge = control_actions.get('battery_discharge', 0)
        
        # Calculate CHP outputs
        chp_heat = chp_power * (self.params["chp"]["eta_th"] / self.params["chp"]["eta_el"])
        chp_fuel = (chp_power + chp_heat) / (self.params["chp"]["eta_el"] + self.params["chp"]["eta_th"])
        
        # Calculate heat pump
        outdoor_temp = weather_data.get('temperature', 10)
        water_temp = 50  # Simplified assumption
        cop = self.calculate_heat_pump_cop(outdoor_temp, water_temp)
        heat_pump_power = heat_pump_heat / cop if cop > 0 else 0
        
        # Calculate electric heater power
        electric_heater_power = electric_heater_heat / self.params["electric_heaters"]["eta"]
        
        # Calculate PV generation
        pv_generation = self.calculate_pv_generation(timestamp, weather_data)
        
        # Battery operations
        battery_params = self.params["battery_storage"]
        if battery_charge > 0:
            actual_charge = min(battery_charge, 
                              battery_params["max_charge_rate"],
                              (battery_params["max_level"] - self.battery_level) / battery_params["charge_efficiency"])
            self.battery_level += actual_charge * battery_params["charge_efficiency"]
            battery_charge_power = actual_charge
        else:
            battery_charge_power = 0
            
        if battery_discharge > 0:
            actual_discharge = min(battery_discharge,
                                 battery_params["max_discharge_rate"],
                                 (self.battery_level - battery_params["min_level"]) * battery_params["discharge_efficiency"])
            self.battery_level -= actual_discharge / battery_params["discharge_efficiency"]
            battery_discharge_power = actual_discharge
        else:
            battery_discharge_power = 0
        
        # Self-discharge
        self.battery_level *= (1 - battery_params["self_discharge_rate"])
        
        # Thermal storage operations
        total_heat_supply = chp_heat + boiler_heat + heat_pump_heat + electric_heater_heat
        heat_balance = total_heat_supply - heat_demand
        
        thermal_params = self.params["thermal_storage"]
        if heat_balance > 0:  # Excess heat
            charge_heat = min(heat_balance, 
                            (thermal_params["max_level"] - self.thermal_storage_level) / thermal_params["charge_efficiency"])
            self.thermal_storage_level += charge_heat * thermal_params["charge_efficiency"]
            discharge_heat = 0
        else:  # Heat deficit
            discharge_heat = min(-heat_balance,
                               (self.thermal_storage_level - thermal_params["min_level"]) * thermal_params["discharge_efficiency"])
            self.thermal_storage_level -= discharge_heat / thermal_params["discharge_efficiency"]
            charge_heat = 0
        
        # Electricity balance
        total_electricity_consumption = (chp_power + heat_pump_power + electric_heater_power + 
                                       battery_charge_power)
        total_electricity_generation = pv_generation + battery_discharge_power
        grid_import = max(0, total_electricity_consumption - total_electricity_generation)
        grid_export = max(0, total_electricity_generation - total_electricity_consumption)
        
        # Economic calculations
        fuel_cost = chp_fuel * self.params["economics"]["fuel_price"]
        boiler_fuel_cost = boiler_heat / self.params["boiler"]["eta"] * self.params["economics"]["fuel_price"]
        electricity_cost = grid_import * electricity_price
        electricity_revenue = grid_export * self.params["economics"]["electricity_price_sell"]
        chp_revenue = chp_power * electricity_price
        
        total_cost = fuel_cost + boiler_fuel_cost + electricity_cost
        total_revenue = chp_revenue + electricity_revenue
        profit = total_revenue - total_cost
        
        # Store results
        result = {
            'timestamp': timestamp,
            'heat_demand': heat_demand,
            'chp_power': chp_power,
            'chp_heat': chp_heat,
            'boiler_heat': boiler_heat,
            'heat_pump_heat': heat_pump_heat,
            'heat_pump_power': heat_pump_power,
            'electric_heater_heat': electric_heater_heat,
            'electric_heater_power': electric_heater_power,
            'pv_generation': pv_generation,
            'battery_charge': battery_charge_power,
            'battery_discharge': battery_discharge_power,
            'battery_level': self.battery_level,
            'thermal_storage_charge': charge_heat,
            'thermal_storage_discharge': discharge_heat,
            'thermal_storage_level': self.thermal_storage_level,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'electricity_price': electricity_price,
            'fuel_cost': fuel_cost,
            'boiler_fuel_cost': boiler_fuel_cost,
            'electricity_cost': electricity_cost,
            'electricity_revenue': electricity_revenue,
            'chp_revenue': chp_revenue,
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'profit': profit
        }
        
        self.results.append(result)
        return result
    
    def run_simulation(self, heat_demand_forecast: pd.DataFrame, 
                      electricity_price_forecast: pd.DataFrame,
                      weather_forecast: pd.DataFrame,
                      control_schedule: pd.DataFrame) -> pd.DataFrame:
        """Run complete simulation"""
        print("🚀 Starting extended CHP simulation...")
        
        self.reset_simulation()
        
        # Merge all input data
        df = heat_demand_forecast.merge(electricity_price_forecast, on='timestamp')
        df = df.merge(weather_forecast, on='timestamp')
        df = df.merge(control_schedule, on='timestamp')
        
        for _, row in df.iterrows():
            weather_data = {
                'temperature': row.get('temperature', 10),
                'solar_irradiance': row.get('solar_irradiance', 0),
                'humidity': row.get('humidity', 50)
            }
            
            control_actions = {
                'chp_power': row.get('chp_power', 0),
                'boiler_heat': row.get('boiler_heat', 0),
                'heat_pump_heat': row.get('heat_pump_heat', 0),
                'electric_heater_heat': row.get('electric_heater_heat', 0),
                'battery_charge': row.get('battery_charge', 0),
                'battery_discharge': row.get('battery_discharge', 0)
            }
            
            self.simulate_timestep(
                timestamp=row['timestamp'],
                heat_demand=row['heat_demand'],
                electricity_price=row['electricity_price'],
                weather_data=weather_data,
                control_actions=control_actions
            )
        
        results_df = pd.DataFrame(self.results)
        
        # Save results
        os.makedirs("data", exist_ok=True)
        results_df.to_csv("data/simulation_results.csv", index=False)
        
        # Print summary
        total_profit = results_df['profit'].sum()
        total_heat_supplied = (results_df['chp_heat'] + results_df['boiler_heat'] + 
                              results_df['heat_pump_heat'] + results_df['electric_heater_heat']).sum()
        total_pv_generation = results_df['pv_generation'].sum()
        
        print(f"✅ Simulation completed!")
        print(f"💰 Total profit: €{total_profit:.2f}")
        print(f"🔥 Total heat supplied: {total_heat_supplied:.1f} kWh")
        print(f"☀️ Total PV generation: {total_pv_generation:.1f} kWh")
        
        return results_df
    
    def run_simulation_from_files(self, parameters_file: str = "data/parameters.json",
                                 demand_file: str = "data/heat_demand_forecast.csv",
                                 price_file: str = "data/electricity_price_forecast.csv",
                                 weather_file: str = "data/weather_forecast.csv",
                                 control_file: str = "data/control_schedule.csv"):
        """Run simulation using data from files"""
        try:
            # Load all required files
            heat_demand = pd.read_csv(demand_file, parse_dates=['timestamp'])
            electricity_price = pd.read_csv(price_file, parse_dates=['timestamp'])
            
            # Try to load weather data, create default if not available
            try:
                weather = pd.read_csv(weather_file, parse_dates=['timestamp'])
            except FileNotFoundError:
                print("⚠️ Weather file not found, using default weather data")
                weather = self._generate_default_weather(heat_demand['timestamp'])
            
            # Try to load control schedule, create default if not available
            try:
                control = pd.read_csv(control_file, parse_dates=['timestamp'])
            except FileNotFoundError:
                print("⚠️ Control file not found, using default control schedule")
                control = self._generate_default_control(heat_demand['timestamp'])
            
            return self.run_simulation(heat_demand, electricity_price, weather, control)
            
        except Exception as e:
            print(f"❌ Error running simulation from files: {e}")
            return None
    
    def _generate_default_weather(self, timestamps: pd.Series) -> pd.DataFrame:
        """Generate default weather data"""
        weather_data = []
        for timestamp in timestamps:
            hour = timestamp.hour
            # Simple weather model
            temperature = 15 + 10 * np.sin(2 * np.pi * (hour - 6) / 24)  # Daily cycle
            solar_irradiance = max(0, 800 * np.sin(2 * np.pi * (hour - 6) / 24))  # Solar cycle
            humidity = 50 + 20 * np.random.random()
            
            weather_data.append({
                'timestamp': timestamp,
                'temperature': temperature,
                'solar_irradiance': solar_irradiance,
                'humidity': humidity
            })
        
        return pd.DataFrame(weather_data)
    
    def _generate_default_control(self, timestamps: pd.Series) -> pd.DataFrame:
        """Generate default control schedule"""
        control_data = []
        for timestamp in timestamps:
            hour = timestamp.hour
            # Simple control strategy
            chp_power = 100 if 8 <= hour <= 20 else 50  # Higher during day
            boiler_heat = 50 if hour < 6 or hour > 22 else 0  # Night heating
            heat_pump_heat = 100 if 6 <= hour <= 22 else 0  # Day heating
            electric_heater_heat = 0  # Off by default
            battery_charge = 20 if 10 <= hour <= 16 else 0  # Charge during peak sun
            battery_discharge = 20 if 18 <= hour <= 22 else 0  # Discharge during peak demand
            
            control_data.append({
                'timestamp': timestamp,
                'chp_power': chp_power,
                'boiler_heat': boiler_heat,
                'heat_pump_heat': heat_pump_heat,
                'electric_heater_heat': electric_heater_heat,
                'battery_charge': battery_charge,
                'battery_discharge': battery_discharge
            })
        
        return pd.DataFrame(control_data)

# Legacy function for backward compatibility
def run_simulation_from_files(parameters_file="data/parameters.json", 
                             demand_file="data/heat_demand_forecast.csv",
                             price_file="data/electricity_price_forecast.csv",
                             control_file="data/control_schedule.csv"):
    """Legacy function for backward compatibility"""
    simulation = ExtendedCHPSimulation(parameters_file)
    return simulation.run_simulation_from_files(
        parameters_file, demand_file, price_file, 
        "data/weather_forecast.csv", control_file
    )

# Example usage
if __name__ == "__main__":
    # Initialize simulation
    simulation = ExtendedCHPSimulation()
    
    # Generate sample data for testing
    timestamps = pd.date_range(start=datetime.now(), periods=24, freq='H')
    
    heat_demand = pd.DataFrame({
        'timestamp': timestamps,
        'heat_demand': 200 + 50 * np.sin(2 * np.pi * timestamps.hour / 24)
    })
    
    electricity_price = pd.DataFrame({
        'timestamp': timestamps,
        'electricity_price': 0.25 + 0.1 * np.sin(2 * np.pi * timestamps.hour / 24)
    })
    
    weather = simulation._generate_default_weather(timestamps)
    control = simulation._generate_default_control(timestamps)
    
    # Run simulation
    results = simulation.run_simulation(heat_demand, electricity_price, weather, control)
    
    print("\n📊 Simulation Results Summary:")
    print(f"Total profit: €{results['profit'].sum():.2f}")
    print(f"Average battery level: {results['battery_level'].mean():.1f} kWh")
    print(f"Average thermal storage: {results['thermal_storage_level'].mean():.1f} kWh")

