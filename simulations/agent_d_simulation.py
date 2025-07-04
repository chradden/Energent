import pandas as pd
import json
import os

def run_simulation_from_files(parameters_file="data/parameters.json", 
                             demand_file="data/heat_demand_forecast.csv",
                             price_file="data/electricity_price_forecast.csv",
                             control_file="data/control_schedule.csv"):
    """
    Führt eine Simulation basierend auf Dateien durch.
    Diese Funktion wird nur aufgerufen, wenn die Dateien existieren.
    """
    try:
        # Lade Parameter
        with open(parameters_file) as f:
            params = json.load(f)

        # Lade Zeitreihen
        df_demand = pd.read_csv(demand_file, parse_dates=["timestamp"])
        df_price = pd.read_csv(price_file, parse_dates=["timestamp"])
        df_control = pd.read_csv(control_file, parse_dates=["timestamp"])

        # Merge alles
        df = df_demand.merge(df_price, on="timestamp").merge(df_control, on="timestamp")

        # Init
        storage = [params["initial_storage"]]
        results = []

        for i, row in df.iterrows():
            P_chp = row["action_bhkw"] * params["P_max"]
            Q_chp = P_chp * (params["eta_th"] / params["eta_el"])
            Q_boiler = row["action_boiler"] * params["Boiler_max"]

            supply = Q_chp + Q_boiler
            demand = row["heat_demand"]
            delta = supply - demand

            if delta >= 0:
                Q_charge = min(delta, params["Storage_capacity"] - storage[-1])
                Q_discharge = 0
                new_storage = storage[-1] + Q_charge
            else:
                Q_discharge = min(-delta, storage[-1])
                Q_charge = 0
                new_storage = storage[-1] - Q_discharge

            fuel_chp = (P_chp + Q_chp) / (params["eta_el"] + params["eta_th"])
            fuel_boiler = Q_boiler / params["eta_boiler"]
            fuel_cost = (fuel_chp + fuel_boiler) * row["gas_price"]
            revenue = P_chp * row["power_price"]
            profit = revenue - fuel_cost

            results.append({
                "timestamp": row["timestamp"],
                "P_chp": P_chp,
                "Q_chp": Q_chp,
                "Q_boiler": Q_boiler,
                "Q_charge": Q_charge,
                "Q_discharge": Q_discharge,
                "storage": new_storage,
                "revenue": revenue,
                "fuel_cost": fuel_cost,
                "profit": profit,
            })
            storage.append(new_storage)

        df_result = pd.DataFrame(results)
        
        # Erstelle output Ordner falls nicht vorhanden
        os.makedirs("output", exist_ok=True)
        df_result.to_csv("output/simulation_results.csv", index=False)
        
        return df_result
        
    except FileNotFoundError as e:
        print(f"Simulation-Dateien nicht gefunden: {e}")
        return None
    except Exception as e:
        print(f"Fehler bei der Simulation: {e}")
        return None

# Diese Datei enthält nur Funktionen und wird nicht sofort ausgeführt

