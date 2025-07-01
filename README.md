# ENERGENT


**Intelligent Multi-Agent System for Optimizing Combined Heat and Power (CHP) Operation**

ENERGENT is a modular, AI-powered system for economic optimization of CHP plants with thermal storage and auxiliary boilers. It combines forecasting, optimization, and simulation in a transparent and responsible way – aligned with NGD’s AI strategy and GDPR requirements.

---

## 🚦 Architecture Overview

Agent A ──→ Heat Demand Forecast

Agent B ──→ Electricity Price Forecast

Agent C ──→ Optimization Engine (LP/MILP or RL)

Agent D ──→ Simulation Environment (Digital Twin)

↓

Streamlit Dashboard


---

## 🤖 Agents Description

### 🔹 Agent A – Heat Forecast
Predicts hourly heat demand using weather data and historic consumption. Implements models like:
- LSTM (PyTorch)
- Temporal Fusion Transformer (TFT)
- XGBoost / Prophet (as fallback)

### 🔹 Agent B – Electricity Price Forecast
Forecasts hourly EPEX spot prices using:
- LSTM / GRU
- Transformer
- External API access (e.g. ENTSO-E Transparency Platform)

### 🔹 Agent C – Optimization Agent
Creates a 24h optimal dispatch plan using:
- Linear/Mixed Integer Programming (`Pyomo`, `PuLP`)
- Reinforcement Learning (PPO with `Stable-Baselines3`)
- Considers BHKW, boiler, thermal storage, fuel prices, and electricity market

### 🔹 Agent D – Simulation Environment
Digital twin of the CHP system. Evaluates any schedule by:
- Simulating thermal and electrical flows
- Tracking storage state, fuel use, revenues
- Providing feedback/rewards for learning agents

---

## 📁 Repository Structure

EBERGENT/

├── agents/

│ ├── agent_a_heat_forecast.py

│ ├── agent_b_price_forecast.py

│ ├── agent_c_optimizer_lp.py

│ ├── agent_c_optimizer_rl.py

├── simulation/

│ └── agent_d_simulation.py

├── data/

│ └── sample_profiles/ # heat_demand.csv, power_price.csv

├── dashboard/

│ └── app.py # Streamlit interface

├── notebooks/

│ └── development and experiments

├── requirements.txt

└── README.md



---

## 🚀 Quickstart

### 1. Clone the repository


git clone https://github.com/chradden/energent.git

cd energent


### 2. Install dependencies

python -m venv venv

source venv/bin/activate  # or .\venv\Scripts\activate on Windows

pip install -r requirements.txt


### 3. Run the Streamlit dashboard

streamlit run dashboard/app.py


---

## 📊 Data Sources
Heat demand: Synthetic or open datasets (e.g. Cornell CHP, Danish Smart Meter data)

Electricity prices: ENTSO-E Transparency Platform or EPEX Spot

Weather data: Open-Meteo, DWD, Meteostat

## ✅ Compliance & Ethics
This project follows the NGD AI Policy and is aligned with:

GDPR (EU General Data Protection Regulation)

Ethical AI development principles

Human-in-the-loop decision architecture

## 📄 License
This project is open-sourced under the MIT License (or specify your license). See LICENSE for details.

## 🤝 Contributors
Developed by energy management and AI teams at Norddeutsche Gesellschaft für Diakonie (NGD), supported by external open-source communities.

##   🧭 Vision
We believe in responsible AI for sustainable energy systems in the social and healthcare sector. ENERGENT empowers local facilities to run smarter, greener, and more cost-efficient.
