import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="smartENERGY API Explorer", page_icon="⚡", layout="centered")

st.title("⚡ smartENERGY API Explorer")
st.write("Abruf von Strompreisdaten für beliebige Zeiträume und Regionen")

# Eingabeparameter
region = st.selectbox("Region", ["AT", "DE", "CH", "FR", "IT"])
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Startdatum", datetime.now() - timedelta(days=1))
with col2:
    end_date = st.date_input("Enddatum", datetime.now())

resolution = st.selectbox("Auflösung", ["hourly", "quarter-hourly"])

if st.button("Daten abrufen"):
    with st.spinner("Hole Daten von smartENERGY API..."):
        url = "https://apis.smartenergy.at/market/v1/price"
        headers = {
            'User-Agent': 'ENERGENT-Explorer/1.0',
            'Accept': 'application/json'
        }
        params = {
            'region': region,
            'start_date': start_date.strftime("%Y-%m-%d"),
            'end_date': end_date.strftime("%Y-%m-%d"),
            'resolution': resolution
        }
        try:
            response = requests.get(url, headers=headers, params=params, timeout=20)
            if response.status_code == 200:
                data = response.json()
                # Versuche, die Daten als DataFrame zu parsen
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict) and "data" in data:
                    df = pd.DataFrame(data["data"])
                else:
                    st.error("Unbekanntes API-Format!")
                    st.json(data)
                    df = None
                if df is not None and not df.empty:
                    st.success(f"{len(df)} Datensätze geladen.")
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("CSV herunterladen", csv, f"smartenergy_{region}_{start_date}_{end_date}.csv", "text/csv")
                else:
                    st.warning("Keine Daten für den gewählten Zeitraum gefunden.")
            else:
                st.error(f"API-Fehler: {response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error(f"Fehler beim API-Abruf: {e}")