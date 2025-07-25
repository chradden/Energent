import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta, date
import json
import pytz

# Add parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_a_heat_forecast import AgentAHeatForecast
from agents.agent_b_price_forecast import AgentBPriceForecast
from agents.agent_c_optimizer_lp import *
from agents.agent_c_optimizer_rl import *
from simulations.agent_d_simulation import *

# PV-Forecast importieren
from agents.PV_forecast import PVForecaster

# Electricity Forecast importieren
from agents.agent_d_electricity_forecast import get_electricity_forecast

# Add import for new XGBoost electricity agent
def try_import_agent_e():
    try:
        from agents.agent_e_electricity_forecast import AgentEElectricityForecast
        return AgentEElectricityForecast
    except ImportError:
        return None
AgentEElectricityForecast = try_import_agent_e()

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
# Ensure prices_dynamic is a numpy array of float
prices_dynamic = np.asarray(prices_dynamic, dtype=float)

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

# Sidebar: Koordinaten für PV-Anlage und Elektrizitätsprognose
st.sidebar.header("Standort & Prognose")
pv_lat = st.sidebar.number_input("Breitengrad (Latitude)", min_value=-90.0, max_value=90.0, value=54.3233, step=0.0001, format="%.4f")
pv_lon = st.sidebar.number_input("Längengrad (Longitude)", min_value=-180.0, max_value=180.0, value=10.1228, step=0.0001, format="%.4f")

# Elektrizitätsprognose Parameter
st.sidebar.header("Elektrizitätsprognose")
electricity_csv_path = st.sidebar.text_input(
    "CSV-Pfad für Elektrizitätsdaten", 
    value="data/electricity consumption_2024-01-01.csv",
    help="Pfad zur CSV-Datei mit historischen Elektrizitätsverbrauchsdaten"
)

# PV-Vorhersage berechnen und visualisieren
st.header("☀️ PV-Leistungsvorhersage für morgen")
try:
    pv_forecaster = PVForecaster(latitude=pv_lat, longitude=pv_lon)
    orig_file_path = "data/historical_Data/PV-electricity_2024_01_01.csv"  # Passe ggf. an
    prep_file_path = "data/historical_Data/PV-prepared.csv"
    # Datenpräparation, falls nötig
    if not os.path.exists(prep_file_path):
        st.info("PV-Daten werden vorbereitet...")
        pv_forecaster.prepare_data(orig_file_path)
        st.success("PV-Datenpräparation abgeschlossen.")
    # Daten laden und Modell trainieren
    pv_forecaster.read_prepared_data(prep_file_path)
    pv_forecaster.train_xgboost()
    st.info(f"Modellgüte (MSE): {pv_forecaster.model_mse}")
    pv_pred = pv_forecaster.predict_next_day()
    # Zeitachse für morgen generieren
    cet = pytz.timezone('Europe/Berlin')
    tomorrow = (datetime.now(cet) + timedelta(days=1)).date()
    timestamps = [datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=h) for h in range(24)]
    # Tabelle mit PV-Vorhersagewerten erzeugen
    import pandas as pd
    df_pv = pd.DataFrame({
        'Zeit': timestamps,
        'PV-Leistung (kW)': pv_pred
    })
    st.dataframe(df_pv, use_container_width=True)

    # Add a line chart for PV forecast
    import plotly.graph_objs as go
    # Ensure Zeit is datetime
    df_pv['Zeit'] = pd.to_datetime(df_pv['Zeit'])
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scatter(
        x=df_pv['Zeit'],
        y=df_pv['PV-Leistung (kW)'],
        mode='lines+markers',
        name='PV-Leistung (kW)',
        line=dict(color='orange', width=2),
        marker=dict(size=6)
    ))
    fig_pv.update_layout(
        title='PV-Leistungsvorhersage für morgen',
        xaxis_title='Zeit',
        yaxis_title='PV-Leistung (kW)',
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig_pv, use_container_width=True)
except Exception as e:
    st.error(f"Fehler bei der PV-Vorhersage: {e}")

# --- HEAT DEMAND FORECAST (Agent A) ---
if (
    'forecasts' not in st.session_state
    or st.session_state['forecasts'] is None
    or not isinstance(st.session_state['forecasts'], dict)
    or 'heat' not in st.session_state['forecasts']
):
    st.info('Now running heat forecast (Agent A)...')
    heat_csv_path = 'data/historical_Data/Gas usage combined_2024-01-01s.csv'
    agent_a = AgentAHeatForecast(model_type='lstm')
    agent_a.try_load_or_train(heat_csv_path, lat=pv_lat, lon=pv_lon)
    heat_forecast_df = agent_a.predict_next_7_days(lat=pv_lat, lon=pv_lon)
    heat_forecast = heat_forecast_df['heat_demand_forecast'].tolist()
    heat_timestamps = heat_forecast_df['timestamp'].tolist()
    st.session_state['forecasts'] = {}  # Always set to dict here
    st.session_state['forecasts']['heat'] = heat_forecast
    st.session_state['forecasts']['heat_timestamps'] = heat_timestamps
    st.session_state['forecasts']['timestamps'] = heat_timestamps  # Use same timestamps for now
    st.session_state['forecasts']['price'] = [0.25] * len(heat_forecast)  # Placeholder price

# --- PRICE FORECAST (Agent B) ---
if (
    'forecasts' not in st.session_state
    or st.session_state['forecasts'] is None
    or not isinstance(st.session_state['forecasts'], dict)
    or 'price' not in st.session_state['forecasts']
    or st.session_state['forecasts']['price'] == [0.25] * len(st.session_state['forecasts']['heat'])  # Check if it's still placeholder
):
    st.info('Now running price forecast (Agent B)...')
    agent_b = AgentBPriceForecast()
    cet = pytz.timezone('Europe/Berlin')
    now = datetime.now(cet)
    tomorrow = (now + timedelta(days=1)).date()
    price_data = agent_b.get_prices_for_day(tomorrow)
    
    if 'date' in price_data.columns and 'value' in price_data.columns:
        price_data = price_data.rename(columns={'date': 'timestamp', 'value': 'price'})
    
    if not price_data.empty and 'timestamp' in price_data.columns and 'price' in price_data.columns:
        price_data = price_data.sort_values('timestamp')
        price_forecast = price_data['price'].tolist()
        price_timestamps = [pd.to_datetime(ts) for ts in price_data['timestamp'].tolist()]
        
        # Update forecasts with real price data
        st.session_state['forecasts']['price'] = price_forecast
        st.session_state['forecasts']['timestamps'] = price_timestamps
        st.session_state['forecasts']['price_timestamps'] = price_timestamps
    else:
        st.warning("Could not fetch price data, using placeholder values")

# Electricity consumption forecast (XGBoost only)
st.header("🔌 Electricity Consumption Forecast")
try:
    with st.spinner("Electricity consumption forecast for the next 7 days is being calculated..."):
        if AgentEElectricityForecast is not None:
            agent_e = AgentEElectricityForecast()
            agent_e.train_from_csv(electricity_csv_path, lat=pv_lat, lon=pv_lon)
            forecast_df, _ = agent_e.predict_next_7_days(lat=pv_lat, lon=pv_lon)
            model_used = "XGBoost"
        else:
            forecast_df = pd.DataFrame()
            model_used = "Unknown"
    if not forecast_df.empty:
        st.success(f"✅ Electricity consumption forecast ({model_used}) successfully calculated!")
        # Plotly-Visualisierung
        fig_electricity = go.Figure()
        fig_electricity.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['electricity_consumption_forecast'],
            mode='lines+markers',
            name='Forecasted Consumption (kWh)',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
        fig_electricity.update_layout(
            title=f"Electricity Consumption Forecast ({model_used})",
            xaxis_title="Time",
            yaxis_title="Forecasted Consumption (kWh)",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_electricity, use_container_width=True)
        # New: Bar chart for average electricity consumption per day (starting from tomorrow)
        st.subheader("Average Electricity Consumption Per Day (Next 7 Days)")
        forecast_df_reset = forecast_df.reset_index()
        forecast_df_reset['date'] = forecast_df_reset['timestamp'].dt.date
        # Get tomorrow's date
        tomorrow = date.today() + timedelta(days=1)
        # Filter to only dates from tomorrow onwards
        forecast_df_reset = forecast_df_reset[forecast_df_reset['date'] >= tomorrow]
        daily_avg = forecast_df_reset.groupby('date')['electricity_consumption_forecast'].mean()
        # Bar chart
        import plotly.graph_objects as go
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=daily_avg.index.astype(str),
            y=daily_avg.values,
            marker_color='lightskyblue',
            name='Avg Consumption (kWh)'
        ))
        fig_bar.update_layout(
            title="Average Electricity Consumption Per Day (Next 7 Days)",
            xaxis_title="Date",
            yaxis_title="Average Consumption (kWh)",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        # New: Line chart for next 24 hours electricity forecast (styled like PV forecast)
        st.subheader("Electricity Consumption Forecast (Next 24 Hours)")
        import plotly.graph_objs as go
        # Get the next 24 hours from forecast_df
        forecast_24h = forecast_df.iloc[:24].copy()
        # Format x-axis as hour:minute
        forecast_24h['time'] = forecast_24h.index.strftime('%H:%M')
        fig_24h = go.Figure()
        fig_24h.add_trace(go.Scatter(
            x=forecast_24h['time'],
            y=forecast_24h['electricity_consumption_forecast'],
            mode='lines+markers',
            name='Electricity Consumption (kWh)',
            line=dict(color='green', width=3, shape='spline'),
            marker=dict(size=6)
        ))
        # Set x-ticks every 4 hours
        tickvals = forecast_24h['time'][::4]
        fig_24h.update_layout(
            title='Electricity Consumption Forecast (Next 24 Hours)',
            xaxis_title='Time',
            yaxis_title='Electricity Consumption (kWh)',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray', tickmode='array', tickvals=tickvals),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            showlegend=True,
            legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top')
        )
        st.plotly_chart(fig_24h, use_container_width=True)
    else:
        st.error("❌ Electricity consumption forecast could not be calculated.")
except Exception as e:
    st.error(f"❌ Error in electricity consumption forecast: {str(e)}")
    st.info("💡 Make sure the CSV file exists and is in the correct format.")


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
                           battery_capacity, battery_charge_rate, pv_peak_power, electric_heater_max, heat_pump_max, storage_loss,
                           pv_lat, pv_lon, electricity_csv_path)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 System Overview")
        
        # Agent status cards
        col1_1, col1_2, col1_3, col1_4, col1_5, col1_6, col1_7, col1_8, col1_9 = st.columns(9)
        
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
                <h4>Agent E</h4>
                <p>Electricity Consumption</p>
                <div class="agent-status status-active">Active</div>
            </div>
            """, unsafe_allow_html=True)
        with col1_6:
            st.markdown("""
            <div class="metric-card">
                <h4>Battery</h4>
                <p>Storage</p>
                <div class="agent-status status-active">Configured</div>
            </div>
            """, unsafe_allow_html=True)
        with col1_7:
            st.markdown("""
            <div class="metric-card">
                <h4>PV</h4>
                <p>Photovoltaic</p>
                <div class="agent-status status-active">Configured</div>
            </div>
            """, unsafe_allow_html=True)
        with col1_8:
            st.markdown("""
            <div class="metric-card">
                <h4>Heat Pump</h4>
                <p>Air-Water</p>
                <div class="agent-status status-active">Configured</div>
            </div>
            """, unsafe_allow_html=True)
        with col1_9:
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
                    battery_capacity, battery_charge_rate, pv_peak_power, electric_heater_max, heat_pump_max, storage_loss,
                    pv_lat, pv_lon, electricity_csv_path):
    """Run the complete optimization pipeline"""
    
    with st.spinner("🔄 Running optimization pipeline..."):
        
        # Step 1: Get forecasts from session state
        st.info("Step 1: Using pre-generated forecasts...")
        
        # Get forecasts from session state
        forecasts = st.session_state['forecasts']
        heat_forecast = forecasts['heat']
        price_forecast = forecasts['price']
        pv_forecast = forecasts.get('pv', [0]*24)
        electricity_forecast = forecasts.get('electricity', [0]*24)
        weather_data = forecasts.get('weather', [{}]*24)
        
        # Step 2: Run optimization
        st.info("Step 2: Running optimization...")
        
        if opt_method == "Linear Programming":
            results = run_extended_lp_optimization(
                heat_forecast, price_forecast, pv_forecast, electricity_forecast, weather_data,
                chp_power_max, chp_heat_max, boiler_max, storage_capacity, fuel_price,
                battery_capacity, battery_charge_rate, pv_peak_power, electric_heater_max, heat_pump_max, storage_loss
            )
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

def run_extended_lp_optimization(heat_demand, elec_price, pv_generation, electricity_demand, weather_data,
                                chp_power_max, chp_heat_max, boiler_max, storage_capacity, fuel_price,
                                battery_capacity, battery_charge_rate, pv_peak_power, electric_heater_max, heat_pump_max, storage_loss):
    """Run Extended Linear Programming optimization using ExtendedCHPOptimizer"""
    
    try:
        # Initialize the ExtendedCHPOptimizer
        optimizer = ExtendedCHPOptimizer()
        
        # Create parameters dictionary for the optimizer
        # Note: The optimizer will use its own parameters from the JSON file, but we can override some
        # The optimizer expects specific parameter structure from its JSON file
        
        # Run optimization with all forecasts
        results = optimizer.optimize_24h_schedule(
            heat_demand=heat_demand,
            electricity_prices=elec_price,
            pv_generation=pv_generation,
            weather_data=weather_data
        )
        
        if results['status'] == 'optimal':
            st.success("✅ Extended optimization completed successfully!")
            
            # Convert results to the expected format for the dashboard
            return {
                'method': 'Extended Linear Programming',
                'total_profit': results['summary']['total_profit'],
                'total_revenue': results['summary']['total_revenue'],
                'total_cost': results['summary']['total_cost'],
                'chp_power': results['chp_power'],
                'chp_heat': results['chp_heat'],
                'boiler_heat': results['boiler_heat'],
                'storage': results['thermal_level'],
                'charge': results['thermal_charge'],
                'discharge': results['thermal_discharge'],
                'pv_generation': results['pv_generation'] if 'pv_generation' in results else pv_generation,
                'battery_level': results['battery_level'],
                'battery_charge': results['battery_charge'],
                'battery_discharge': results['battery_discharge'],
                'heat_pump_usage': results['heat_pump_heat'],
                'electric_heater_usage': results['electric_heater_heat'],
                'storage_losses': [storage_loss] * 24,
                'grid_import': results['grid_import'],
                'grid_export': results['grid_export'],
                'chp_on': results['chp_on'],
                'boiler_on': results['boiler_on'],
                'heat_pump_on': results['heat_pump_on'],
                'objective_value': results['objective_value'],
                'status': 'optimal'
            }
        else:
            st.error(f"❌ Extended optimization failed: {results.get('error', 'Unknown error')}")
            # Fallback to simplified optimization
            return run_simplified_lp_optimization(
                heat_demand, elec_price, chp_power_max, chp_heat_max, boiler_max, 
                storage_capacity, fuel_price, battery_capacity, battery_charge_rate, 
                pv_peak_power, electric_heater_max, heat_pump_max, storage_loss
            )
            
    except Exception as e:
        st.error(f"❌ Error in extended optimization: {str(e)}")
        st.info("🔄 Falling back to simplified optimization...")
        # Fallback to simplified optimization
        return run_simplified_lp_optimization(
            heat_demand, elec_price, chp_power_max, chp_heat_max, boiler_max, 
            storage_capacity, fuel_price, battery_capacity, battery_charge_rate, 
            pv_peak_power, electric_heater_max, heat_pump_max, storage_loss
        )

def run_simplified_lp_optimization(heat_demand, elec_price, chp_power_max, chp_heat_max, 
                       boiler_max, storage_capacity, fuel_price,
                       battery_capacity, battery_charge_rate, pv_peak_power, electric_heater_max, heat_pump_max, storage_loss):
    """Run simplified Linear Programming optimization (fallback)"""
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
        'method': 'Simplified Linear Programming (Fallback)',
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

def run_lp_optimization(heat_demand, elec_price, chp_power_max, chp_heat_max, 
                       boiler_max, storage_capacity, fuel_price,
                       battery_capacity, battery_charge_rate, pv_peak_power, electric_heater_max, heat_pump_max, storage_loss):
    """Legacy function - now redirects to extended optimization"""
    return run_extended_lp_optimization(
        heat_demand, elec_price, [], [], [],  # Empty PV, electricity demand, weather for backward compatibility
        chp_power_max, chp_heat_max, boiler_max, storage_capacity, fuel_price,
        battery_capacity, battery_charge_rate, pv_peak_power, electric_heater_max, heat_pump_max, storage_loss
    )

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
    
    # Create 2x2 grid for all forecasts
    col1, col2 = st.columns(2)
    
    with col1:
        # Heat demand forecast
        fig_heat = go.Figure()
        fig_heat.add_trace(go.Scatter(
            x=forecasts['heat_timestamps'],
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
        
        # PV generation forecast
        if 'pv' in forecasts:
            fig_pv = go.Figure()
            timestamps = forecasts.get('price_timestamps', list(range(24)))
            fig_pv.add_trace(go.Scatter(
                x=timestamps,
                y=forecasts['pv'],
                mode='lines+markers',
                name='PV Generation',
                line=dict(color='orange', width=2)
            ))
            fig_pv.update_layout(
                title="24-Hour PV Generation Forecast",
                xaxis_title="Time",
                yaxis_title="PV Generation (kW)",
                height=400
            )
            st.plotly_chart(fig_pv, use_container_width=True)
    
    with col2:
        # Electricity price forecast
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=forecasts.get('price_timestamps', forecasts.get('timestamps', list(range(24)))),
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
        
        # Electricity consumption forecast
        if 'electricity' in forecasts:
            fig_elec = go.Figure()
            timestamps = forecasts.get('price_timestamps', list(range(24)))
            fig_elec.add_trace(go.Scatter(
                x=timestamps,
                y=forecasts['electricity'],
                mode='lines+markers',
                name='Electricity Consumption',
                line=dict(color='green', width=2)
            ))
            fig_elec.update_layout(
                title="24-Hour Electricity Consumption Forecast",
                xaxis_title="Time",
                yaxis_title="Electricity Consumption (kWh)",
                height=400
            )
            st.plotly_chart(fig_elec, use_container_width=True)

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
            x=st.session_state.forecasts.get('price_timestamps', st.session_state.forecasts.get('timestamps', list(range(24)))),
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
            x=st.session_state.forecasts.get('price_timestamps', st.session_state.forecasts.get('timestamps', list(range(24)))),
            y=results['chp_heat'],
            mode='lines+markers',
            name='CHP Heat',
            line=dict(color='orange', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=st.session_state.forecasts.get('price_timestamps', st.session_state.forecasts.get('timestamps', list(range(24)))),
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
            x=st.session_state.forecasts.get('price_timestamps', st.session_state.forecasts.get('timestamps', list(range(24)))),
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
        display_component_time_series(results, st.session_state.forecasts.get('price_timestamps', st.session_state.forecasts.get('timestamps', list(range(24)))))

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
