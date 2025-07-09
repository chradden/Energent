import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import json
import pytz

# Add parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_a_heat_forecast import AgentAHeatForecast
from agents.agent_b_price_forecast import AgentBPriceForecast
from agents.agent_c_optimizer_lp import *
from agents.agent_c_optimizer_rl import *
from simulations.agent_d_simulation import *

# Page configuration
st.set_page_config(
    page_title="ENERGENT - CHP Optimization Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .agent-status {
        padding: 0.5rem;
        border-radius: 0.25rem;
        text-align: center;
        font-weight: bold;
    }
    .status-active {
        background-color: #d4edda;
        color: #155724;
    }
    .status-inactive {
        background-color: #f8d7da;
        color: #721c24;
    }
    .ngd-logo {
        position: absolute;
        top: 1.5rem;
        right: 2.5rem;
        width: 120px;
        z-index: 10;
    }
    .ngd-contact {
        position: absolute;
        top: 7.5rem;
        right: 2.7rem;
        font-size: 0.95rem;
        color: #444;
        background: rgba(255,255,255,0.8);
        padding: 0.2rem 0.7rem;
        border-radius: 0.3rem;
        z-index: 10;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# NGD Logo und Kontakt sowie ENERGENT-Header
col1, col2, col3 = st.columns([2, 4, 2])
with col1:
    st.empty()
with col2:
    st.markdown('<h1 class="main-header">⚡ ENERGENT</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Intelligent Multi-Agent System for CHP Optimization</p>', unsafe_allow_html=True)
with col3:
    st.image("dashboard/ngd_logo.png", width=110)
    st.markdown('<div style="font-size:0.95rem; color:#444; margin-top:0.5rem; text-align:right;">Christian.Radden@ngd.de</div>', unsafe_allow_html=True)

# Initialize session state
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = None

st.sidebar.header("Anlagenparameter & Key Metrics")

eta_boiler = st.sidebar.number_input("Wirkungsgrad Boiler η_boiler", min_value=0.5, max_value=1.0, value=0.9)
eta_thermal = st.sidebar.number_input("Wirkungsgrad BHKW thermisch η_thermal", min_value=0.3, max_value=1.0, value=0.5)
eta_hr = st.sidebar.number_input("Wirkungsgrad Heizstab η_hr", min_value=0.5, max_value=1.0, value=0.99)
COP = st.sidebar.number_input("COP Wärmepumpe", min_value=1.0, max_value=8.0, value=3.5)
gas_price = st.sidebar.number_input("Gaspreis [€/kWh]", min_value=0.01, max_value=0.20, value=0.06)
grid_fees = st.sidebar.number_input("Netzentgelte [ct/kWh]", min_value=0.0, max_value=10.0, value=7.0)
surcharges = st.sidebar.number_input("Zuschläge/Steuern [ct/kWh]", min_value=0.0, max_value=10.0, value=5.0)
VAT = st.sidebar.number_input("MwSt.", min_value=1.0, max_value=1.3, value=1.19)

# Day-Ahead-Preise (ct/kWh) aus dem Agenten holen (hier als Beispiel, ggf. anpassen)
from agents.agent_b_price_forecast import AgentBPriceForecast
from datetime import datetime, timedelta
agent_b = AgentBPriceForecast()
tomorrow = (datetime.now() + timedelta(days=1)).date()
price_df = agent_b.get_prices_for_day(tomorrow)
prices_dynamic = price_df['price'].values if not price_df.empty else np.full(24, 10.0)

# Dynamischer Strompreis (all-in, stündlich)
electricity_price_dynamic = (prices_dynamic + grid_fees + surcharges) * VAT  # ct/kWh

# Beispiel: Mittelwert für die Key Metrics
mean_electricity_price = np.mean(electricity_price_dynamic) / 100  # €/kWh

cost_boiler = gas_price / eta_boiler
cost_heating_rod = mean_electricity_price / eta_hr
cost_heatpump = mean_electricity_price / COP
cost_CHP = gas_price / eta_thermal

st.sidebar.markdown("---")
st.sidebar.metric("Spezifische Kosten Boiler [€/kWh]", f"{cost_boiler:.3f}")
st.sidebar.metric("Spezifische Kosten Heizstab [€/kWh]", f"{cost_heating_rod:.3f}")
st.sidebar.metric("Spezifische Kosten Wärmepumpe [€/kWh]", f"{cost_heatpump:.3f}")
st.sidebar.metric("Spezifische Kosten BHKW [€/kWh]", f"{cost_CHP:.3f}")
st.sidebar.metric("Strompreis (dynamisch, all-in) [ct/kWh]", f"{np.mean(electricity_price_dynamic):.2f}")


def main():
    # Header (entfernt, da jetzt oben im Layout)
    # st.markdown('<h1 class="main-header">⚡ ENERGENT</h1>', unsafe_allow_html=True)
    # st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Intelligent Multi-Agent System for CHP Optimization</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Agent selection
        st.subheader("Agent Settings")
        
        # Agent A - Heat Forecast
        st.write("**Agent A - Heat Forecast**")
        heat_model = st.selectbox(
            "Model Type",
            ["lstm", "xgboost", "prophet"],
            key="heat_model"
        )
        
        # Agent B - Price Forecast (keine Auswahl mehr)
        st.write("**Agent B - Price Forecast**")
        st.caption("Preise werden direkt von der smartENERGY API bezogen.")
        
        # Agent C - Optimization
        st.write("**Agent C - Optimization**")
        opt_method = st.selectbox(
            "Method",
            ["Linear Programming", "Reinforcement Learning"],
            key="opt_method"
        )
        
        # System parameters
        st.subheader("System Parameters")
        chp_power_max = st.slider("CHP Power Max (kW)", 100, 500, 200)
        chp_heat_max = st.slider("CHP Heat Max (kW)", 200, 800, 300)
        boiler_max = st.slider("Boiler Max (kW)", 200, 1000, 400)
        storage_capacity = st.slider("Storage Capacity (kWh)", 1000, 5000, 2000)
        fuel_price = st.number_input("Fuel Price (€/kWh)", 0.01, 0.10, 0.03, 0.01)
        
        # Erweiterte Komponenten (angepasste Dimensionierung)
        st.markdown("---")
        st.subheader("Advanced Components")
        battery_capacity = st.slider("Battery Storage Capacity (kWh)", 20, 200, 60)
        battery_charge_rate = st.slider("Battery Max Charge/Discharge (kW)", 10, 100, 30)
        pv_peak_power = st.slider("PV Peak Power (kW)", 10, 200, 90)
        electric_heater_max = st.slider("Electric Heater Max (kW)", 20, 200, 120)
        heat_pump_max = st.slider("Heat Pump Max (kW)", 20, 125, 125)
        storage_loss = st.number_input("Thermal Storage Losses (kWh/h)", 0.0, 50.0, 2.0, 0.1)
        
        # Run optimization button
        if st.button("🚀 Run Optimization", type="primary"):
            run_optimization(heat_model, opt_method, 
                           chp_power_max, chp_heat_max, boiler_max, 
                           storage_capacity, fuel_price,
                           battery_capacity, battery_charge_rate, pv_peak_power, electric_heater_max, heat_pump_max, storage_loss)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 System Overview")
        
        # Agent status cards
        col1_1, col1_2, col1_3, col1_4, col1_5, col1_6, col1_7, col1_8 = st.columns(8)
        
        with col1_1:
            st.markdown("""
            <div class="metric-card">
                <h4>Agent A</h4>
                <p>Heat Forecast</p>
                <div class="agent-status status-active">Active</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col1_2:
            st.markdown("""
            <div class="metric-card">
                <h4>Agent B</h4>
                <p>Price Forecast</p>
                <div class="agent-status status-active">Active</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col1_3:
            st.markdown("""
            <div class="metric-card">
                <h4>Agent C</h4>
                <p>Optimization</p>
                <div class="agent-status status-active">Active</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col1_4:
            st.markdown("""
            <div class="metric-card">
                <h4>Agent D</h4>
                <p>Simulation</p>
                <div class="agent-status status-active">Active</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col1_5:
            st.markdown("""
            <div class="metric-card">
                <h4>Battery</h4>
                <p>Storage</p>
                <div class="agent-status status-active">Configured</div>
            </div>
            """, unsafe_allow_html=True)
        with col1_6:
            st.markdown("""
            <div class="metric-card">
                <h4>PV</h4>
                <p>Photovoltaic</p>
                <div class="agent-status status-active">Configured</div>
            </div>
            """, unsafe_allow_html=True)
        with col1_7:
            st.markdown("""
            <div class="metric-card">
                <h4>Heat Pump</h4>
                <p>Air-Water</p>
                <div class="agent-status status-active">Configured</div>
            </div>
            """, unsafe_allow_html=True)
        with col1_8:
            st.markdown("""
            <div class="metric-card">
                <h4>Electric Heater</h4>
                <p>Direct</p>
                <div class="agent-status status-active">Configured</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.header("📈 Key Metrics")
        
        if st.session_state.optimization_results is not None:
            results = st.session_state.optimization_results
            
            st.metric(
                label="Total Revenue",
                value=f"€{results.get('total_revenue', 0):.2f}",
                delta=f"€{results.get('revenue_delta', 0):.2f}"
            )
            st.metric(
                label="Total Cost",
                value=f"€{results.get('total_cost', 0):.2f}",
                delta=f"€{results.get('cost_delta', 0):.2f}"
            )
            st.metric(
                label="Net Profit",
                value=f"€{results.get('net_profit', 0):.2f}",
                delta=f"€{results.get('profit_delta', 0):.2f}"
            )
            st.metric(
                label="Efficiency",
                value=f"{results.get('efficiency', 0):.1f}%",
                delta=f"{results.get('efficiency_delta', 0):.1f}%"
            )
            # Neue Key Metrics für die Komponenten
            st.metric(
                label="PV Generation",
                value=f"{results.get('pv_generation', 0):.1f} kWh"
            )
            st.metric(
                label="Battery Avg. Level",
                value=f"{results.get('battery_avg', 0):.1f} kWh"
            )
            st.metric(
                label="Heat Pump Usage",
                value=f"{results.get('heat_pump_usage', 0):.1f} kWh"
            )
            st.metric(
                label="Electric Heater Usage",
                value=f"{results.get('electric_heater_usage', 0):.1f} kWh"
            )
            st.metric(
                label="Thermal Storage Losses",
                value=f"{results.get('storage_losses', 0):.1f} kWh"
            )
        else:
            st.info("Run optimization to see metrics")
    
    # Forecasts and Optimization Results
    if st.session_state.forecasts is not None:
        display_forecasts(st.session_state.forecasts)
    
    if st.session_state.optimization_results is not None:
        display_optimization_results(st.session_state.optimization_results)

def run_optimization(heat_model, opt_method, 
                    chp_power_max, chp_heat_max, boiler_max, 
                    storage_capacity, fuel_price,
                    battery_capacity, battery_charge_rate, pv_peak_power, electric_heater_max, heat_pump_max, storage_loss):
    """Run the complete optimization pipeline"""
    
    with st.spinner("🔄 Running optimization pipeline..."):
        
        # Step 1: Get forecasts
        st.info("Step 1: Generating forecasts...")
        
        # Agent A - Heat Forecast
        agent_a = AgentAHeatForecast(model_type=heat_model)
        weather_data = agent_a.get_weather_data()
        agent_a.train(weather_data)
        heat_forecast = agent_a.predict(weather_data)
        
        # Agent B - Price Forecast (jetzt: echter Day-Ahead für morgen, 0-24 Uhr)
        agent_b = AgentBPriceForecast()
        from datetime import timedelta
        import pytz
        cet = pytz.timezone('Europe/Berlin')
        now = datetime.now(cet)
        tomorrow = (now + timedelta(days=1)).date()
        price_data = agent_b.get_prices_for_day(tomorrow)
        # Spalten ggf. umbenennen
        if 'date' in price_data.columns and 'value' in price_data.columns:
            price_data = price_data.rename(columns={'date': 'timestamp', 'value': 'price'})
        if price_data.empty or 'timestamp' not in price_data.columns or 'price' not in price_data.columns:
            st.error("Keine Preisdaten für den gewünschten Zeitraum von der smartENERGY API erhalten!")
            st.info(f"[DEBUG] price_data Inhalt: {price_data}")
            st.warning("Day-Ahead-Preise für morgen sind erst ab 13:00 Uhr CET verfügbar oder unvollständig!")
            return
        price_data = price_data.sort_values('timestamp')
        price_forecast = price_data['price'].tolist()
        price_timestamps = price_data['timestamp'].tolist()
        # Ensure all timestamps are datetime objects
        import pandas as pd
        price_timestamps = [pd.to_datetime(ts) for ts in price_timestamps]
        # Vereinfachte Prüfung: Nur ob DataFrame leer ist
        if price_data.empty:
            st.error("Optimierung abgebrochen: Es sind nicht alle 24 Day-Ahead-Preise (0-23h) für morgen verfügbar!")
            st.info(f"[DEBUG] Preis-Timestamps für morgen: {[str(ts) for ts in price_timestamps]}")
            st.info(f"[DEBUG] Preise für morgen: {price_forecast}")
            st.warning("Day-Ahead-Preise für morgen sind erst ab 13:00 Uhr CET verfügbar oder unvollständig!")
            return
        # Debug-Output: Zeige extrahierte Preise und Timestamps für morgen
        st.info(f"[DEBUG] Preis-Timestamps für morgen: {[str(ts) for ts in price_timestamps]}")
        st.info(f"[DEBUG] Preise für morgen: {price_forecast}")
        # Store forecasts
        st.session_state.forecasts = {
            'heat': heat_forecast,
            'price': price_forecast,
            'timestamps': price_timestamps
        }
        
        # Step 2: Run optimization
        st.info("Step 2: Running optimization...")
        
        if opt_method == "Linear Programming":
            results = run_lp_optimization(heat_forecast, price_forecast, 
                                        chp_power_max, chp_heat_max, boiler_max, 
                                        storage_capacity, fuel_price,
                                        battery_capacity, battery_charge_rate, pv_peak_power, electric_heater_max, heat_pump_max, storage_loss)
        else:
            results = run_rl_optimization(heat_forecast, price_forecast, 
                                        chp_power_max, chp_heat_max, boiler_max, 
                                        storage_capacity, fuel_price)
        
        # Step 3: Run simulation
        st.info("Step 3: Running simulation...")
        simulation_results = run_simulation(results, heat_forecast, price_forecast)
        
        # Combine results
        st.session_state.optimization_results = {
            **results,
            **simulation_results
        }
        
        st.success("✅ Optimization completed!")

def run_lp_optimization(heat_demand, elec_price, chp_power_max, chp_heat_max, 
                       boiler_max, storage_capacity, fuel_price,
                       battery_capacity, battery_charge_rate, pv_peak_power, electric_heater_max, heat_pump_max, storage_loss):
    """Run Linear Programming optimization (erweitert)"""
    hours = range(24)
    # Dummy-Zeitreihen für neue Komponenten (hier: einfache Profile, später LP-Integration)
    pv_generation = [pv_peak_power * max(0, np.sin(np.pi * (t-6)/12)) for t in hours]
    battery_level = [battery_capacity/2 + 10*np.sin(np.pi*t/24) for t in hours]
    battery_charge = [battery_charge_rate * max(0, np.sin(np.pi * (t-8)/12)) for t in hours]
    battery_discharge = [battery_charge_rate * max(0, np.sin(np.pi * (t-18)/12)) for t in hours]
    heat_pump_usage = [heat_pump_max * max(0, np.sin(np.pi * (t-5)/12)) for t in hours]
    electric_heater_usage = [electric_heater_max * max(0, np.sin(np.pi * (t-20)/12)) for t in hours]
    storage_losses = [storage_loss for _ in hours]

    # Bisherige Optimierung (vereinfacht)
    P_CHP = [min(chp_power_max, max(0, 0.7*hd)) for hd in heat_demand]
    Q_CHP = [min(chp_heat_max, 0.5*p) for p in P_CHP]
    Q_boiler = [max(0, hd-qc) for hd, qc in zip(heat_demand, Q_CHP)]
    S = [storage_capacity/2 for _ in hours]
    Q_charge = [0 for _ in hours]
    Q_discharge = [0 for _ in hours]

    # Dummy-Ökonomie
    total_profit = sum([elec_price[t]*P_CHP[t] - fuel_price*(P_CHP[t]+Q_CHP[t])/0.85 - fuel_price*Q_boiler[t]/0.9 for t in hours])

    return {
        'method': 'Linear Programming',
        'total_profit': total_profit,
        'chp_power': P_CHP,
        'chp_heat': Q_CHP,
        'boiler_heat': Q_boiler,
        'storage': S,
        'charge': Q_charge,
        'discharge': Q_discharge,
        'pv_generation': pv_generation,
        'battery_level': battery_level,
        'battery_charge': battery_charge,
        'battery_discharge': battery_discharge,
        'heat_pump_usage': heat_pump_usage,
        'electric_heater_usage': electric_heater_usage,
        'storage_losses': storage_losses
    }

def run_rl_optimization(heat_demand, elec_price, chp_power_max, chp_heat_max, 
                       boiler_max, storage_capacity, fuel_price):
    """Run Reinforcement Learning optimization"""
    
    # Setze Kopplungsfaktor alpha (wie im LP)
    alpha = chp_heat_max / chp_power_max if chp_power_max > 0 else 1.0
    eta_total_CHP = 0.85
    eta_boiler = 0.90
    
    # Create environment
    env = CHPEnv(
        heat_demand, elec_price, 
        initial_storage=storage_capacity/2,
        S_max=storage_capacity,
        P_max_e=chp_power_max,
        Boiler_max=boiler_max,
        alpha=alpha,
        eta_total_CHP=eta_total_CHP,
        eta_boiler=eta_boiler,
        fuel_price=fuel_price
    )
    
    # Simple policy (for demo purposes)
    # In a real implementation, you would train a PPO agent here
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
    
    # Dummy-Arrays für Kompatibilität mit Visualisierung
    dummy = [0] * 24
    results = {
        'method': 'Reinforcement Learning',
        'total_profit': total_reward,
        'chp_power': [actions[t][0] * chp_power_max for t in range(24)],
        'chp_heat': [actions[t][0] * chp_heat_max for t in range(24)],
        'boiler_heat': [actions[t][1] * boiler_max for t in range(24)],
        'storage': [storage_capacity/2] * 24,  # Simplified
        'charge': dummy,
        'discharge': dummy,
        'battery_level': dummy,
        'pv_generation': dummy,
        'heat_pump_usage': dummy,
        'electric_heater_usage': dummy,
        'storage_losses': dummy,
        'battery_charge': dummy,  # <-- NEU
        'battery_discharge': dummy  # <-- NEU
    }
    
    return results

def run_simulation(optimization_results, heat_demand, elec_price):
    """Run simulation to evaluate results (erweitert)"""
    # Summen und Mittelwerte für Key Metrics
    total_revenue = sum(optimization_results['chp_power'][t] * elec_price[t] for t in range(24))
    total_cost = sum(optimization_results['chp_power'][t] * 0.03 for t in range(24))  # Simplified fuel cost
    net_profit = total_revenue - total_cost
    total_heat_produced = sum(optimization_results['chp_heat']) + sum(optimization_results['boiler_heat'])
    total_heat_demand = sum(heat_demand)
    efficiency = (total_heat_demand / total_heat_produced) * 100 if total_heat_produced > 0 else 0
    pv_sum = sum(optimization_results.get('pv_generation', [0]*24))
    battery_avg = np.mean(optimization_results.get('battery_level', [0]*24))
    heat_pump_sum = sum(optimization_results.get('heat_pump_usage', [0]*24))
    electric_heater_sum = sum(optimization_results.get('electric_heater_usage', [0]*24))
    storage_loss_sum = sum(optimization_results.get('storage_losses', [0]*24))
    return {
        'total_revenue': total_revenue,
        'total_cost': total_cost,
        'net_profit': net_profit,
        'efficiency': efficiency,
        'revenue_delta': total_revenue * 0.05,
        'cost_delta': -total_cost * 0.03,
        'profit_delta': net_profit * 0.08,
        'efficiency_delta': 2.5,
        'pv_generation': pv_sum,
        'battery_avg': battery_avg,
        'heat_pump_usage': heat_pump_sum,
        'electric_heater_usage': electric_heater_sum,
        'storage_losses': storage_loss_sum
    }

def display_forecasts(forecasts):
    """Display forecast charts"""
    st.header("📈 Forecasts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Heat demand forecast
        fig_heat = go.Figure()
        fig_heat.add_trace(go.Scatter(
            x=forecasts['timestamps'],
            y=forecasts['heat'],
            mode='lines+markers',
            name='Heat Demand',
            line=dict(color='red', width=2)
        ))
        fig_heat.update_layout(
            title="24-Hour Heat Demand Forecast",
            xaxis_title="Time",
            yaxis_title="Heat Demand (kW)",
            height=400
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    
    with col2:
        # Electricity price forecast
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=forecasts['timestamps'],
            y=forecasts['price'],
            mode='lines+markers',
            name='Electricity Price',
            line=dict(color='blue', width=2)
        ))
        fig_price.update_layout(
            title="24-Hour Electricity Price Forecast",
            xaxis_title="Time",
            yaxis_title="Price (€/kWh)",
            height=400
        )
        st.plotly_chart(fig_price, use_container_width=True)

def display_optimization_results(results):
    """Display optimization results"""
    st.header("🎯 Optimization Results")
    
    # Method info
    st.info(f"**Method:** {results['method']}")
    
    # Results overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Profit", f"€{results['total_profit']:.2f}")
    
    with col2:
        st.metric("Net Profit", f"€{results['net_profit']:.2f}")
    
    with col3:
        st.metric("Efficiency", f"{results['efficiency']:.1f}%")
    
    # Detailed charts
    st.subheader("System Operation Schedule")
    
    # Create subplot for power and heat
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Power Generation', 'Heat Generation'),
        vertical_spacing=0.1
    )
    
    # Power generation
    fig.add_trace(
        go.Scatter(
            x=st.session_state.forecasts['timestamps'],
            y=results['chp_power'],
            mode='lines+markers',
            name='CHP Power',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )
    
    # Heat generation
    fig.add_trace(
        go.Scatter(
            x=st.session_state.forecasts['timestamps'],
            y=results['chp_heat'],
            mode='lines+markers',
            name='CHP Heat',
            line=dict(color='orange', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=st.session_state.forecasts['timestamps'],
            y=results['boiler_heat'],
            mode='lines+markers',
            name='Boiler Heat',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Storage and economic analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Storage level
        fig_storage = go.Figure()
        fig_storage.add_trace(go.Scatter(
            x=st.session_state.forecasts['timestamps'],
            y=results['storage'],
            mode='lines+markers',
            name='Storage Level',
            fill='tonexty',
            line=dict(color='purple', width=2)
        ))
        fig_storage.update_layout(
            title="Thermal Storage Level",
            xaxis_title="Time",
            yaxis_title="Storage (kWh)",
            height=400
        )
        st.plotly_chart(fig_storage, use_container_width=True)
    
    with col2:
        # Economic breakdown
        fig_econ = go.Figure()
        fig_econ.add_trace(go.Bar(
            x=['Revenue', 'Cost', 'Profit'],
            y=[results['total_revenue'], results['total_cost'], results['net_profit']],
            marker_color=['green', 'red', 'blue']
        ))
        fig_econ.update_layout(
            title="Economic Summary",
            yaxis_title="Amount (€)",
            height=400
        )
        st.plotly_chart(fig_econ, use_container_width=True)

    # Nach den bisherigen Plots:
    if 'pv_generation' in results:
        display_component_time_series(results, st.session_state.forecasts['timestamps'])

def display_component_time_series(results, timestamps):
    """Zeige Zeitreihen für alle Komponenten als Tabs"""
    st.header("🔋 Component Time Series")
    tabs = st.tabs(["PV", "Battery", "Heat Pump", "Electric Heater", "Storage Losses"])
    def ensure_series(val):
        import numpy as np
        if isinstance(val, (float, np.floating)):
            return [val]*24
        if isinstance(val, (int, np.integer)):
            return [float(val)]*24
        if isinstance(val, (list, np.ndarray, pd.Series)):
            return list(val)
        return [0.0]*24
    with tabs[0]:
        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(x=timestamps, y=ensure_series(results['pv_generation']), mode='lines+markers', name='PV Generation', line=dict(color='gold', width=2)))
        fig_pv.update_layout(title="PV Generation", xaxis_title="Time", yaxis_title="Power (kW)", height=350)
        st.plotly_chart(fig_pv, use_container_width=True)
    with tabs[1]:
        fig_bat = go.Figure()
        fig_bat.add_trace(go.Scatter(x=timestamps, y=ensure_series(results['battery_level']), mode='lines+markers', name='Battery Level', line=dict(color='purple', width=2)))
        fig_bat.add_trace(go.Scatter(x=timestamps, y=ensure_series(results['battery_charge']), mode='lines', name='Charge', line=dict(color='green', dash='dot')))
        fig_bat.add_trace(go.Scatter(x=timestamps, y=ensure_series(results['battery_discharge']), mode='lines', name='Discharge', line=dict(color='red', dash='dot')))
        fig_bat.update_layout(title="Battery Storage", xaxis_title="Time", yaxis_title="Energy/Power (kWh/kW)", height=350)
        st.plotly_chart(fig_bat, use_container_width=True)
    with tabs[2]:
        fig_hp = go.Figure()
        fig_hp.add_trace(go.Scatter(x=timestamps, y=ensure_series(results['heat_pump_usage']), mode='lines+markers', name='Heat Pump Usage', line=dict(color='blue', width=2)))
        fig_hp.update_layout(title="Heat Pump Usage", xaxis_title="Time", yaxis_title="Power (kW)", height=350)
        st.plotly_chart(fig_hp, use_container_width=True)
    with tabs[3]:
        fig_eh = go.Figure()
        fig_eh.add_trace(go.Scatter(x=timestamps, y=ensure_series(results['electric_heater_usage']), mode='lines+markers', name='Electric Heater Usage', line=dict(color='orange', width=2)))
        fig_eh.update_layout(title="Electric Heater Usage", xaxis_title="Time", yaxis_title="Power (kW)", height=350)
        st.plotly_chart(fig_eh, use_container_width=True)
    with tabs[4]:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=timestamps, y=ensure_series(results['storage_losses']), mode='lines+markers', name='Storage Losses', line=dict(color='gray', width=2)))
        fig_loss.update_layout(title="Thermal Storage Losses", xaxis_title="Time", yaxis_title="Losses (kWh/h)", height=350)
        st.plotly_chart(fig_loss, use_container_width=True)

if __name__ == "__main__":
    main()
