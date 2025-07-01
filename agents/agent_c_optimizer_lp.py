import pulp

# Zeitindex und bekannte Parameter (Beispielwerte)
hours = range(24)
P_max_e = 200.0   # kW elektrische BHKW-Nennleistung
Q_max_th = 300.0  # kW thermische BHKW-Nennleistung
Boiler_max = 400.0  # kW Kesselkapazität
S_max = 2000.0    # kWh Speicherkapazität (entspricht z.B. ~30 m³ Wasser:contentReference[oaicite:25]{index=25})
eta_total_CHP = 0.85
eta_boiler = 0.90
alpha = Q_max_th / P_max_e  # Wärme-Strom-Faktor, hier 1.5

# Beispiel-Prognosedaten als Listen (24 Werte)
demand = [...]       # z.B. [250, 230, 220, ..., 300] kW Wärmebedarf pro Stunde
elec_price = [...]   # z.B. [0.05, 0.04, ..., 0.12] €/kWh Strompreis pro Stunde
fuel_price = 0.03    # €/kWh Brennstoff (Erdgas)

# LP-Modell initialisieren
model = pulp.LpProblem("CHP_schedule", pulp.LpMaximize)

# Variablen definieren
P_CHP = pulp.LpVariable.dicts("P_CHP_elec", hours, lowBound=0, upBound=P_max_e)
Q_CHP = pulp.LpVariable.dicts("Q_CHP_heat", hours, lowBound=0, upBound=Q_max_th)
Q_boiler = pulp.LpVariable.dicts("Q_boiler", hours, lowBound=0, upBound=Boiler_max)
Q_charge = pulp.LpVariable.dicts("Q_charge", hours, lowBound=0)
Q_discharge = pulp.LpVariable.dicts("Q_discharge", hours, lowBound=0)
S = pulp.LpVariable.dicts("Storage", hours, lowBound=0, upBound=S_max)

# Nebenbedingungen: 
for t in hours:
    # KWK-Kopplung: Wärme = alpha * Strom
    model += Q_CHP[t] == alpha * P_CHP[t]
    # Wärmebilanz: Erzeugung + Entladung = Bedarf + Ladung
    model += Q_CHP[t] + Q_boiler[t] + Q_discharge[t] == demand[t] + Q_charge[t]
    # Speicherbilanz:
    if t == 0:
        model += S[0] == S_max / 2  # z.B. halbvoll zu Beginn
    else:
        model += S[t] == S[t-1] + Q_charge[t-1] - Q_discharge[t-1]

# Zielfunktion: Summe der stündlichen Gewinne (Erlös - Kosten)
total_profit = pulp.lpSum([
    elec_price[t] * P_CHP[t] 
    - fuel_price * (P_CHP[t] + Q_CHP[t]) / eta_total_CHP   # Brennstoffkosten BHKW
    - fuel_price * Q_boiler[t] / eta_boiler               # Brennstoffkosten Kessel
    for t in hours
])
model += total_profit

# LP maximieren:
model.solve(pulp.PULP_CBC_CMD(msg=False))
print("Optimales Ergebnis:", pulp.value(total_profit))
for t in hours:
    print(t, P_CHP[t].value(), Q_boiler[t].value(), S[t].value())

