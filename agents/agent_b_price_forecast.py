import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta

class AgentBPriceForecast:
    """Agent B: Electricity Price Forecasting (minimal)"""
    def __init__(self, model_type: str = "transformer"):
        self.model_type = model_type

    def get_entsoe_data(self, date, api_key, country_code='DE_LU') -> pd.DataFrame:
        """
        Holt Day-Ahead-Preise für den angegebenen Tag von der ENTSO-E Transparency Platform.
        Gibt ein DataFrame mit 'timestamp' (Europe/Berlin, volle Stunde) und 'price' (ct/kWh) zurück.
        Benötigt das Paket entsoe-py und einen gültigen API-Key.
        """
        from entsoe import EntsoePandasClient
        import pandas as pd
        import pytz
        tz = pytz.timezone('Europe/Berlin')
        client = EntsoePandasClient(api_key=api_key)
        start = pd.Timestamp(date, tz=tz)
        end = start + pd.Timedelta(days=1)
        try:
            prices = client.query_day_ahead_prices(country_code, start=start, end=end)
            # Umrechnung in ct/kWh
            prices_ct = prices / 10
            df = prices_ct.reset_index()
            df.columns = ['timestamp', 'price']
            return df
        except Exception as e:
            print(f"[ERROR] ENTSO-E API: {e}")
            return pd.DataFrame(columns=["timestamp", "price"])

    def get_epexspot_scrape(self, trading_date, delivery_date) -> pd.DataFrame:
        """
        Scrape EPEX Spot Day-Ahead Preise für DE-LU von der offiziellen Market Results Seite.
        Gibt ein DataFrame mit 'timestamp' (Europe/Berlin, volle Stunde) und 'price' (ct/kWh) zurück.
        """
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        import pytz

        url = (
            f"https://www.epexspot.com/en/market-results?"
            f"market_area=DE-LU&auction=MRC&trading_date={trading_date}&delivery_date={delivery_date}"
            f"&modality=Auction&sub_modality=DayAhead&data_mode=table"
        )
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")
            if table is None:
                print("[ERROR] Keine Preistabelle auf der EPEX-Seite gefunden!")
                print("[DEBUG] HTML-Response:", response.text[:1000])
                return pd.DataFrame(columns=["timestamp", "price"])
            rows = table.find_all("tr")
            prices = []
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 4:
                    price = cols[-1].text.strip().replace(",", ".")
                    try:
                        price = float(price)
                        prices.append(price)
                    except:
                        continue
            # Erzeuge Timestamps
            tz = pytz.timezone('Europe/Berlin')
            timestamps = [tz.localize(pd.Timestamp(f"{delivery_date} {h:02d}:00")) for h in range(24)]
            prices_ct = [p / 10 for p in prices[:24]]  # €/MWh -> ct/kWh
            df = pd.DataFrame({
                "timestamp": timestamps,
                "price": prices_ct
            })
            print("[DEBUG] Gescrapete EPEX-Preise:")
            print(df)
            return df
        except Exception as e:
            print(f"[ERROR] EPEX Spot Scraping: {e}")
            return pd.DataFrame(columns=["timestamp", "price"])

    def get_prices_for_day(self, date, region: str = "DE_LU") -> pd.DataFrame:
        """
        Holt Day-Ahead-Preise für DE-LU für den gewünschten Tag per Web-Scraping von der EPEX Market Results Seite.
        Gibt ein DataFrame mit 24 Stundenwerten (timestamp, price in ct/kWh) zurück.
        Fallback auf synthetische Werte, falls das Scraping fehlschlägt.
        """
        import pandas as pd
        import numpy as np
        import pytz
        from datetime import datetime, timedelta
        # delivery_date = gewünschtes Datum, trading_date = Vortag
        delivery_date = pd.to_datetime(date).strftime("%Y-%m-%d")
        trading_date = (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            df = self.get_epexspot_scrape(trading_date, delivery_date)
            if df is not None and not df.empty and len(df) == 24:
                return df
            else:
                print("[WARN] EPEX Spot Scraping lieferte keine vollständigen Daten, nutze Fallback.")
        except Exception as e:
            print(f"[ERROR] EPEX Spot Scraping: {e}")
        # Fallback: synthetische Werte
        tz = pytz.timezone('Europe/Berlin')
        timestamps = [tz.localize(pd.Timestamp(f"{delivery_date} {h:02d}:00")) for h in range(24)]
        base_price = 10 + 2 * np.sin(2 * np.pi * np.arange(24) / 24)
        volatility = np.random.normal(0, 0.5, 24)
        trend = np.linspace(0, 1, 24)
        price = base_price + volatility + trend
        synth_day = pd.DataFrame({
            'timestamp': timestamps,
            'price': price
        })
        return synth_day.reset_index(drop=True)
