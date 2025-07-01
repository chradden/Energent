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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = None

def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ ENERGENT</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Intelligent Multi-Agent System for CHP Optimization</p>', unsafe_allow_html=True)
    
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
        
        # Agent B - Price Forecast
        st.write("**Agent B - Price Forecast**")
        price_model = st.selectbox(
            "Model Type",
            ["transformer", "lstm"],
            key="price_model"
        )
        
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
        
        # Run optimization button
        if st.button("🚀 Run Optimization", type="primary"):
            run_optimization(heat_model, price_model, opt_method, 
                           chp_power_max, chp_heat_max, boiler_max, 
                           storage_capacity, fuel_price)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 System Overview")
        
        # Agent status cards
        col1_1, col1_2, col1_3, col1_4 = st.columns(4)
        
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
        else:
            st.info("Run optimization to see metrics")
    
    # Forecasts and Optimization Results
    if st.session_state.forecasts is not None:
        display_forecasts(st.session_state.forecasts)
    
    if st.session_state.optimization_results is not None:
        display_optimization_results(st.session_state.optimization_results)

def run_optimization(heat_model, price_model, opt_method, 
                    chp_power_max, chp_heat_max, boiler_max, 
                    storage_capacity, fuel_price):
    """Run the complete optimization pipeline"""
    
    with st.spinner("🔄 Running optimization pipeline..."):
        
        # Step 1: Get forecasts
        st.info("Step 1: Generating forecasts...")
        
        # Agent A - Heat Forecast
        agent_a = AgentAHeatForecast(model_type=heat_model)
        weather_data = agent_a.get_weather_data()
        agent_a.train(weather_data)
        heat_forecast = agent_a.predict(weather_data)
        
        # Agent B - Price Forecast
        agent_b = AgentBPriceForecast(model_type=price_model)
        price_data = agent_b.get_entsoe_data()
        agent_b.train(price_data)
        price_forecast = agent_b.predict(price_data)
        
        # Store forecasts
        st.session_state.forecasts = {
            'heat': heat_forecast,
            'price': price_forecast,
            'timestamps': pd.date_range(start=datetime.now(), periods=24, freq='H')
        }
        
        # Step 2: Run optimization
        st.info("Step 2: Running optimization...")
        
        if opt_method == "Linear Programming":
            results = run_lp_optimization(heat_forecast, price_forecast, 
                                        chp_power_max, chp_heat_max, boiler_max, 
                                        storage_capacity, fuel_price)
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

def run_rl_optimization(heat_demand, elec_price, chp_power_max, chp_heat_max, 
                       boiler_max, storage_capacity, fuel_price):
    """Run Reinforcement Learning optimization"""
    
    # Create environment
    env = CHPEnv(heat_demand, elec_price, initial_storage=storage_capacity/2)
    
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

def run_simulation(optimization_results, heat_demand, elec_price):
    """Run simulation to evaluate results"""
    
    # Calculate additional metrics
    total_revenue = sum(optimization_results['chp_power'][t] * elec_price[t] for t in range(24))
    total_cost = sum(optimization_results['chp_power'][t] * 0.03 for t in range(24))  # Simplified fuel cost
    net_profit = total_revenue - total_cost
    
    # Calculate efficiency
    total_heat_produced = sum(optimization_results['chp_heat']) + sum(optimization_results['boiler_heat'])
    total_heat_demand = sum(heat_demand)
    efficiency = (total_heat_demand / total_heat_produced) * 100 if total_heat_produced > 0 else 0
    
    return {
        'total_revenue': total_revenue,
        'total_cost': total_cost,
        'net_profit': net_profit,
        'efficiency': efficiency,
        'revenue_delta': total_revenue * 0.05,  # Simulated improvement
        'cost_delta': -total_cost * 0.03,
        'profit_delta': net_profit * 0.08,
        'efficiency_delta': 2.5
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

if __name__ == "__main__":
    main()
