import gymnasium as gym
import numpy as np

class CHPEnv(gym.Env):
    def __init__(self, demand_profile, price_profile, initial_storage=1000.0,
                 S_max=2000.0, P_max_e=200.0, Boiler_max=400.0, alpha=1.5,
                 eta_total_CHP=0.85, eta_boiler=0.90, fuel_price=0.03):
        super().__init__()
        self.demand_profile = demand_profile
        self.price_profile = price_profile
        self.T = len(demand_profile)  # z.B. 24
        self.initial_storage = initial_storage
        self.S_max = S_max
        self.P_max_e = P_max_e
        self.Boiler_max = Boiler_max
        self.alpha = alpha
        self.eta_total_CHP = eta_total_CHP
        self.eta_boiler = eta_boiler
        self.fuel_price = fuel_price
        # Beobachtungsraum definieren: z.B. [Stunde, Speicherstand, nächster Preis, nächster Bedarf]
        high = np.array([24.0, self.S_max, max(price_profile)*2, max(demand_profile)*2], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.zeros(4, dtype=np.float32), high=high, dtype=np.float32)
        # Aktionsraum: 2-dim kontinuirlich zwischen 0 und 1 (Anteile)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    
    def reset(self):
        self.t = 0
        self.storage = self.initial_storage
        return self._get_state()
    
    def _get_state(self):
        return np.array([self.t, self.storage, 
                         self.price_profile[self.t], 
                         self.demand_profile[self.t]], dtype=np.float32)
    
    def step(self, action):
        # Action-Scaling
        u_chp, u_boiler = float(action[0]), float(action[1])
        P_chp = u_chp * self.P_max_e
        Q_chp = P_chp * self.alpha
        Q_boiler = u_boiler * self.Boiler_max
        # Wärmebilanz in der Simulation:
        demand = self.demand_profile[self.t]
        supply = Q_chp + Q_boiler
        # Standard: erst Bedarf decken, Überschuss in Speicher, Mangel aus Speicher
        charged = 0.0
        discharged = 0.0
        if supply >= demand:
            # Überschuss – lade Speicher
            charged = min(supply - demand, self.S_max - self.storage)
            self.storage += charged
            unmet = 0.0
        else:
            # Unterdeckung – entlade Speicher
            needed = demand - supply
            discharged = min(needed, self.storage)
            self.storage -= discharged
            unmet = needed - discharged  # falls immer noch Bedarf offen
        # Falls noch unmet > 0 (Speicher leer und supply reichte nicht), 
        # muss Boiler mehr machen (Notfall):
        if unmet > 1e-6:
            # Strafe: sehr hoher negativer Reward oder an Kessel anrechnen
            Q_boiler += unmet
            # (Wir könnten Q_boiler hier erhöhen, aber dann stimmt action vs. state nicht ganz.
            # Alternativ: unmet als negatives Reward-Penalty.)
        # Reward berechnen:
        revenue = P_chp * self.price_profile[self.t]
        fuel_cost_chp = (P_chp + Q_chp) / self.eta_total_CHP * self.fuel_price
        fuel_cost_boiler = Q_boiler / self.eta_boiler * self.fuel_price
        reward = revenue - fuel_cost_chp - fuel_cost_boiler
        # zum nächsten Zeitschritt
        self.t += 1
        done = (self.t >= self.T)
        return self._get_state() if not done else np.zeros(4, dtype=np.float32), reward, done, {}

