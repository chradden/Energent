# ⚡ ENERGENT

**Intelligent Multi-Agent System for Optimizing Combined Heat and Power (CHP) Operation**

ENERGENT is a modular, AI-powered system for economic optimization of CHP plants with thermal storage and auxiliary boilers. It combines forecasting, optimization, and simulation in a transparent and responsible way – aligned with NGD's AI strategy and GDPR requirements.

---

## 🚦 Architecture Overview

```
Agent A ──→ Heat Demand Forecast
Agent B ──→ Electricity Price Forecast  
Agent C ──→ Optimization Engine (LP/MILP or RL)
Agent D ──→ Simulation Environment (Digital Twin)
                ↓
        Streamlit Dashboard
```

---

## 🤖 Agents Description

### 🔹 Agent A – Heat Forecast
Predicts hourly heat demand using weather data and historic consumption. Implements models like:
- **LSTM** (PyTorch) - Deep learning for time series
- **XGBoost** - Gradient boosting for tabular data
- **Prophet** (Facebook) - Statistical forecasting as fallback

### 🔹 Agent B – Electricity Price Forecast
Forecasts hourly EPEX spot prices using:
- **Transformer** - State-of-the-art attention-based model
- **LSTM** - Recurrent neural network
- **External API** access (ENTSO-E Transparency Platform, Yahoo Finance)

### 🔹 Agent C – Optimization Agent
Creates a 24h optimal dispatch plan using:
- **Linear Programming** (`PuLP`) - Mathematical optimization
- **Reinforcement Learning** (PPO with `Stable-Baselines3`) - AI-based optimization
- Considers BHKW, boiler, thermal storage, fuel prices, and electricity market

### 🔹 Agent D – Simulation Environment
Digital twin of the CHP system. Evaluates any schedule by:
- Simulating thermal and electrical flows
- Tracking storage state, fuel use, revenues
- Providing feedback/rewards for learning agents

---

## 📁 Repository Structure

```
ENERGENT/
├── agents/
│   ├── agent_a_heat_forecast.py      # ✅ Complete
│   ├── agent_b_price_forecast.py     # ✅ Complete
│   ├── agent_c_optimizer_lp.py       # ✅ Complete
│   └── agent_c_optimizer_rl.py       # ✅ Complete
├── simulations/
│   └── agent_d_simulation.py         # ✅ Complete
├── dashboard/
│   └── app.py                        # ✅ Complete
├── data/
│   ├── parameters.json               # ✅ Complete
│   ├── heat_demand_forecast.csv      # Generated
│   ├── electricity_price_forecast.csv # Generated
│   └── optimization_results.json     # Generated
├── main.py                           # ✅ Complete
├── requirements.txt                  # ✅ Complete
└── README.md                         # ✅ Complete
```

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/chradden/energent.git
cd energent
```

### 2. Install dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the system

#### Option A: Command Line Interface
```bash
# Run complete pipeline with default settings
python main.py

# Run with custom parameters
python main.py --heat-model lstm --price-model transformer --opt-method linear_programming

# Run with different optimization method
python main.py --opt-method reinforcement_learning
```

#### Option B: Web Dashboard
```bash
# Launch Streamlit dashboard
python main.py --dashboard

# Or directly
streamlit run dashboard/app.py
```

### 4. View Results
- **Forecasts**: `data/heat_demand_forecast.csv`, `data/electricity_price_forecast.csv`
- **Optimization**: `data/optimization_results.json`
- **Simulation**: `data/simulation_results.json`
- **Interactive**: Web dashboard at `http://localhost:8501`

---

## 🎯 Usage Examples

### Basic Optimization Run
```python
from main import ENERGENTOrchestrator

# Initialize system
orchestrator = ENERGENTOrchestrator()

# Run complete pipeline
results = orchestrator.run_complete_pipeline(
    heat_model="lstm",
    price_model="transformer", 
    opt_method="linear_programming"
)

print(f"Total Profit: €{results['optimization']['total_profit']:.2f}")
```

### Custom Configuration
```python
# Load custom parameters
orchestrator = ENERGENTOrchestrator("my_config.json")

# Run with specific models
results = orchestrator.run_complete_pipeline(
    heat_model="xgboost",
    price_model="lstm",
    opt_method="reinforcement_learning"
)
```

### Individual Agent Usage
```python
from agents.agent_a_heat_forecast import AgentAHeatForecast

# Use Agent A directly
agent_a = AgentAHeatForecast(model_type="lstm")
weather_data = agent_a.get_weather_data()
agent_a.train(weather_data)
heat_forecast = agent_a.predict(weather_data)
```

---

## 📊 Data Sources

### Heat Demand
- **Real-time**: Weather API (Open-Meteo)
- **Historical**: Synthetic data generation
- **Future**: Integration with building management systems

### Electricity Prices
- **Real-time**: ENTSO-E Transparency Platform
- **Backup**: Yahoo Finance market data
- **Synthetic**: Realistic price patterns for testing

### Weather Data
- **Primary**: Open-Meteo API (free, no registration)
- **Alternative**: DWD, Meteostat
- **Features**: Temperature, humidity, wind speed, precipitation

---

## ⚙️ Configuration

### System Parameters (`data/parameters.json`)
```json
{
  "system_parameters": {
    "chp": {
      "P_max": 200.0,      // Max electrical power (kW)
      "Q_max": 300.0,      // Max thermal power (kW)
      "eta_el": 0.35,      // Electrical efficiency
      "eta_th": 0.50       // Thermal efficiency
    },
    "boiler": {
      "Q_max": 400.0,      // Max boiler power (kW)
      "eta": 0.90          // Boiler efficiency
    },
    "storage": {
      "capacity": 2000.0,  // Storage capacity (kWh)
      "initial_level": 1000.0
    }
  }
}
```

### Command Line Options
```bash
python main.py --help
# Options:
#   --config CONFIG        Configuration file path
#   --heat-model {lstm,xgboost,prophet}  Heat forecasting model
#   --price-model {transformer,lstm}     Price forecasting model  
#   --opt-method {linear_programming,reinforcement_learning}  Optimization method
#   --dashboard           Launch Streamlit dashboard
```

---

## 📈 Performance Metrics

### Forecasting Accuracy
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)  
- **MAPE** (Mean Absolute Percentage Error)

### Optimization Results
- **Total Profit** (€/day)
- **Revenue** vs **Cost** breakdown
- **Efficiency** (%)
- **CO2 Emissions** (kg)

### System Performance
- **Storage Utilization** (%)
- **CHP vs Boiler** usage ratio
- **Peak vs Off-peak** optimization

---

## 🔧 Development

### Adding New Models
```python
# In agent_a_heat_forecast.py
class NewHeatModel:
    def train(self, data):
        # Your training logic
        pass
    
    def predict(self, data):
        # Your prediction logic
        return forecast

# Add to AgentAHeatForecast class
def train_new_model(self, data):
    self.model = NewHeatModel()
    self.model.train(data)
```

### Extending Optimization
```python
# In agent_c_optimizer_lp.py
# Add new constraints or objective functions
model += new_constraint
model += new_objective_term
```

### Custom Dashboard Pages
```python
# In dashboard/app.py
def new_page():
    st.header("Custom Analysis")
    # Your custom visualizations
```

---

## ✅ Compliance & Ethics

This project follows the NGD AI Policy and is aligned with:

- **GDPR** (EU General Data Protection Regulation)
- **Ethical AI** development principles
- **Human-in-the-loop** decision architecture
- **Transparency** in optimization decisions
- **Responsible** energy management

---

## 🧪 Testing

### Unit Tests
```bash
# Run individual agent tests
python -m pytest tests/test_agent_a.py
python -m pytest tests/test_agent_b.py
python -m pytest tests/test_agent_c.py
```

### Integration Tests
```bash
# Test complete pipeline
python -m pytest tests/test_integration.py
```

### Performance Benchmarks
```bash
# Benchmark optimization methods
python benchmarks/compare_methods.py
```

---

## 🚨 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**API Connection Issues**
```bash
# Check internet connection
# Verify API endpoints are accessible
# Use synthetic data as fallback
```

**Optimization Failures**
```bash
# Check system parameters in data/parameters.json
# Verify input data quality
# Try different optimization methods
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python main.py --debug
```

---

## 📄 License

This project is open-sourced under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributors

Developed by energy management and AI teams at **Norddeutsche Gesellschaft für Diakonie (NGD)**, supported by external open-source communities.

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 🧭 Vision

We believe in **responsible AI for sustainable energy systems** in the social and healthcare sector. ENERGENT empowers local facilities to run smarter, greener, and more cost-efficient.

### Future Roadmap
- [ ] Real-time data integration
- [ ] Multi-site optimization
- [ ] Advanced ML models (TFT, N-BEATS)
- [ ] Mobile app interface
- [ ] API for third-party integration
- [ ] Carbon footprint optimization
- [ ] Predictive maintenance

---

## 📞 Support

- **Documentation**: [Wiki](https://github.com/chradden/energent/wiki)
- **Issues**: [GitHub Issues](https://github.com/chradden/energent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chradden/energent/discussions)
- **Email**: energent@ngd.de

---

**⚡ ENERGENT - Powering the Future of Energy Management**
