import streamlit as st
import pandas as pd
import numpy as np
import glob, os, joblib

st.set_page_config(page_title="Pest Prediction Demo", layout="wide")

st.title("Pest Prediction Demo — Maruteru (Prototype)")
st.markdown(
    """
    This demo loads saved per-pest models (trained with log1p target + lag features)
    and allows you to:

    * Run predictions for a chosen **Standard Week** (uses a historical week as a proxy for weather)
    * Or upload a **single-row weekly CSV** (aggregated daily → weekly) and predict from that

    The goal is to show how weather + recent pest levels map to predicted pest pressure for each pest.
    """
)

# ----------------- Utility functions ----------------- #

@st.cache_data(show_spinner=False)
def load_models(models_dir="models_all"):
    models = {}
    model_files = sorted(glob.glob(os.path.join(models_dir, "*.pkl")))
    for mf in model_files:
        try:
            pack = joblib.load(mf)
            models[pack["pest_name"]] = pack
        except Exception as e:
            st.warning(f"Failed loading {mf}: {e}")
    return models


@st.cache_data(show_spinner=False)
def load_pest_csvs(data_dir="data"):
    pest_dfs = {}
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.replace("\n", " ").strip() for c in df.columns]
            pest_name = (
                os.path.basename(csv_path)
                .replace("Rice_Maruteru - ", "")
                .replace("(Number_ Light trap).csv", "")
                .strip()
            )
            pest_dfs[pest_name] = df
        except Exception as e:
            st.warning(f"Failed reading {csv_path}: {e}")
    return pest_dfs


def build_features_from_row(row: pd.Series, features, df_full=None):
    vals = {}
    for f in features:
        if f in row.index:
            vals[f] = row.get(f, np.nan)
        else:
            vals[f] = np.nan

        # if lag missing, fill with median pest value when possible
        if f in ("pest_lag1", "pest_lag2") and (pd.isna(vals[f]) or vals[f] is None):
            if df_full is not None and "PestValue" in df_full.columns:
                vals[f] = float(df_full["PestValue"].median())
    return pd.DataFrame([vals], columns=features)


def predict_from_pack(pack, Xrow):
    # impute
    imp = pack.get("imputer")
    if imp is not None:
        try:
            X_imp = pd.DataFrame(imp.transform(Xrow), columns=pack["features"])
        except Exception:
            X_imp = Xrow.fillna(0)
    else:
        X_imp = Xrow.fillna(0)

    model = pack["model"]
    if pack.get("transform") == "log1p":
        pred_log = model.predict(X_imp)[0]
        pred = float(np.expm1(pred_log))
    else:
        pred = float(model.predict(X_imp)[0])

    thr = pack.get("thresholds")
    if thr:
        if pred <= 0:
            risk = "None"
        elif pred <= thr["p50"]:
            risk = "Low"
        elif pred <= thr["p75"]:
            risk = "Medium"
        else:
            risk = "High"
    else:
        risk = "High" if pred > 10 else "Low"

    return pred, risk


def risk_badge(risk: str) -> str:
    if risk == "High":
        return "🔴 High"
    if risk == "Medium":
        return "🟡 Medium"
    if risk == "Low":
        return "🟢 Low"
    return "⚪ None"


def extract_weather_snapshot(row: pd.Series, columns) -> dict:
    temp = humid = rain = None
    for c in columns:
        if "MaxT" in c and pd.notna(row.get(c)):
            temp = row.get(c)
        if "RH1" in c and pd.notna(row.get(c)):
            humid = row.get(c)
        if "RF" in c and pd.notna(row.get(c)):
            rain = row.get(c)
    return {"temp": temp, "humid": humid, "rain": rain}


# ----------------- UI layout ----------------- #

models_dir_default = "models_all"
data_dir_default = "data"

models = load_models(models_dir_default)
pest_dfs = load_pest_csvs(data_dir_default)

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Inputs")
    data_dir = st.text_input("Data folder (CSV files)", value=data_dir_default)
    models_dir = st.text_input("Models folder", value=models_dir_default)

    mode = st.radio("Prediction mode", ["Historical (week)", "Upload weekly CSV"])
    if mode == "Historical (week)":
        week = st.slider("Standard Week (1–52)", 1, 52, 10)
    else:
        uploaded = st.file_uploader("Upload single-row weekly CSV", type=["csv"])

    run = st.button("Run Predictions")

with col2:
    st.header("Status")
    if not models:
        st.info("Models not loaded yet. Train them locally with `pest_prediction_pipeline.py` and place .pkl files in `models_all/`.")
    else:
        st.success(f"Loaded {len(models)} pest models.")

st.markdown("---")

# ----------------- Main logic ----------------- #

if run:
    if not models:
        st.error("No models found. Put your trained .pkl files under the 'models_all' folder (or change Models folder).")
    else:
        results = []
        weather_info = None  # for the weather panel

        if mode == "Historical (week)":
            st.subheader(f"Historical proxy mode — Week {week}")
            st.write(
                "For now, we emulate a future forecast by re-using the most recent historical "
                "record around this week at Maruteru."
            )

            # Use the first pest df we have to show a weather snapshot
            first_df = None
            weather_cols_any = []
            if pest_dfs:
                first_df = list(pest_dfs.values())[0]
                weather_cols_any = [c for c in first_df.columns if any(k in c for k in ["MaxT","MinT","RH","RF","WS","SSH","EVP","Temp","Humid","Rain"])]

            for pest_name, pack in models.items():
                df = pest_dfs.get(pest_name)
                if df is None:
                    st.warning(f"No CSV loaded for {pest_name}; cannot build historical row. Skipping.")
                    continue

                df.columns = [c.replace("\n", " ").strip() for c in df.columns]

                # choose a row for that week
                if "Standard Week" in df.columns and (df["Standard Week"] == week).any():
                    row = df[df["Standard Week"] == week].iloc[-1]
                elif "Standard Week" in df.columns and ((df["Standard Week"] % 52) == (week % 52)).any():
                    row = df[(df["Standard Week"] % 52) == (week % 52)].iloc[-1]
                else:
                    row = df.iloc[-1]

                # store weather snapshot from the FIRST pest only
                if weather_info is None and weather_cols_any:
                    weather_info = extract_weather_snapshot(row, weather_cols_any)

                Xrow = build_features_from_row(row, pack["features"], df_full=df)
                pred, risk = predict_from_pack(pack, Xrow)
                year_val = row.get("Observation Year")

                results.append(
                    {
                        "Pest": pest_name,
                        "Predicted Count": float(pred),
                        "Risk": risk,
                        "Year": int(year_val) if pd.notna(year_val) else None,
                    }
                )

        else:  # Upload weekly CSV mode
            st.subheader("Uploaded weekly input mode")
            if not uploaded:
                st.warning("Upload a single-row weekly CSV file to use this mode.")
            else:
                try:
                    wdf = pd.read_csv(uploaded)
                    wdf.columns = [c.replace("\n", " ").strip() for c in wdf.columns]
                    row = wdf.iloc[0]

                    weather_cols_any = [c for c in wdf.columns if any(k in c for k in ["MaxT","MinT","RH","RF","WS","SSH","EVP","Temp","Humid","Rain"])]
                    weather_info = extract_weather_snapshot(row, weather_cols_any)

                    for pest_name, pack in models.items():
                        Xrow = build_features_from_row(row, pack["features"], df_full=wdf)
                        pred, risk = predict_from_pack(pack, Xrow)
                        results.append(
                            {
                                "Pest": pest_name,
                                "Predicted Count": float(pred),
                                "Risk": risk,
                            }
                        )
                except Exception as e:
                    st.error(f"Failed to read uploaded file: {e}")

        # ----- If we have results, format + show them ----- #
        if results:
            resdf = pd.DataFrame(results)

            # Make counts & year nice integers for display
            if "Predicted Count" in resdf.columns:
                resdf["Predicted Count"] = resdf["Predicted Count"].apply(lambda x: int(round(x)))
            if "Year" in resdf.columns:
                resdf["Year"] = resdf["Year"].apply(lambda x: int(x) if pd.notna(x) else None)

            # Add risk badges
            resdf["Risk Badge"] = resdf["Risk"].apply(risk_badge)

            # Sort pests by risk level and count
            risk_order = {"High": 3, "Medium": 2, "Low": 1, "None": 0}
            resdf["__risk_rank"] = resdf["Risk"].map(risk_order)
            resdf = resdf.sort_values(["__risk_rank", "Predicted Count"], ascending=[False, False]).drop(columns="__risk_rank")

            # ========== Weather snapshot panel ========== #
            if weather_info is not None:
                st.markdown("### Weather snapshot used for this prediction")
                c1, c2, c3 = st.columns(3)
                temp_txt = f"{weather_info['temp']:.1f} °C" if weather_info["temp"] is not None else "N/A"
                humid_txt = f"{weather_info['humid']:.1f} %" if weather_info["humid"] is not None else "N/A"
                rain_txt = f"{weather_info['rain']:.1f} mm" if weather_info["rain"] is not None else "N/A"
                c1.metric("Max Temperature", temp_txt)
                c2.metric("Relative Humidity (RH1)", humid_txt)
                c3.metric("Rainfall (RF)", rain_txt)
                st.markdown("---")

            # ========== Results table with colored risk ========== #
            st.markdown("### Pest risk overview")
            display_cols = ["Pest", "Predicted Count", "Risk Badge"]
            if "Year" in resdf.columns:
                display_cols.append("Year")
            st.table(resdf[display_cols])

            # ========== Download as CSV button ========== #
            csv_bytes = resdf.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download results as CSV",
                data=csv_bytes,
                file_name="pest_predictions.csv",
                mime="text/csv",
            )
        else:
            st.info("No predictions produced. Check inputs or upload a weekly CSV.")
