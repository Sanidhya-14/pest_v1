# app.py
import streamlit as st
import pandas as pd
import numpy as np
import glob, os, joblib
from math import exp
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Pest Prediction Demo", layout="wide")

st.title("Pest Prediction Demo — Maruteru (Prototype)")
st.markdown(
    """
    This demo loads saved per-pest models (trained with log1p target + lag features)
    and allows you to:
    * Run predictions for a chosen Standard Week (uses historical row as proxy for weather)
    * Upload a single-row weekly CSV (aggregated daily -> weekly) and predict from it
    """
)

@st.cache_data(show_spinner=False)
def load_models(models_dir="models_all"):
    models = {}
    model_files = sorted(glob.glob(os.path.join(models_dir, "*.pkl")))
    for mf in model_files:
        try:
            pack = joblib.load(mf)
            models[pack['pest_name']] = pack
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
            df.columns = [c.replace("\n"," ").strip() for c in df.columns]
            pest_name = os.path.basename(csv_path).replace('Rice_Maruteru - ', '').replace('(Number_ Light trap).csv','').strip()
            pest_dfs[pest_name] = df
        except Exception as e:
            st.warning(f"Failed reading {csv_path}: {e}")
    return pest_dfs

def build_features_from_row(row, features, df_full=None):
    # row: pd.Series of the weekly input (may be historical row)
    vals = {}
    for f in features:
        if f in row.index:
            vals[f] = row.get(f, np.nan)
        else:
            vals[f] = np.nan
        if f in ('pest_lag1', 'pest_lag2') and (pd.isna(vals[f]) or vals[f] is None):
            # fallback median from df_full
            if df_full is not None and 'PestValue' in df_full.columns:
                vals[f] = float(df_full['PestValue'].median())
    return pd.DataFrame([vals], columns=features)

def predict_from_pack(pack, Xrow):
    imp = pack.get('imputer')
    if imp is not None:
        try:
            X_imp = pd.DataFrame(imp.transform(Xrow), columns=pack['features'])
        except Exception:
            # if transform fails, fallback to fillna
            X_imp = Xrow.fillna(0)
    else:
        X_imp = Xrow.fillna(0)
    model = pack['model']
    if pack.get('transform') == 'log1p':
        pred_log = model.predict(X_imp)[0]
        pred = float(np.expm1(pred_log))
    else:
        pred = float(model.predict(X_imp)[0])
    thr = pack.get('thresholds')
    if thr:
        if pred <= 0:
            risk = 'None'
        elif pred <= thr['p50']:
            risk = 'Low'
        elif pred <= thr['p75']:
            risk = 'Medium'
        else:
            risk = 'High'
    else:
        risk = 'High' if pred > 10 else 'Low'
    return pred, risk

# UI controls
col1, col2 = st.columns([1,2])

with col1:
    st.header("Inputs")
    data_dir = st.text_input("Local data folder (CSV files)", value="data")
    models_dir = st.text_input("Models folder", value="models_all")
    mode = st.radio("Prediction mode", ["Historical (week)","Upload weekly CSV"])
    if mode == "Historical (week)":
        week = st.slider("Standard Week (1-52)", 1, 52, 10)
    else:
        uploaded = st.file_uploader("Upload single-row weekly CSV", type=["csv"])
    run = st.button("Run Predictions")

with col2:
    st.header("Status / Info")
    st.write("Models and CSVs will be loaded from the given folders (relative to repo).")

models = load_models(models_dir)
pest_dfs = load_pest_csvs(data_dir)

if run:
    if not models:
        st.error("No models found. Put your saved models under the 'models_all' folder (or change Models folder).")
    else:
        results = []
        if mode == "Historical (week)":
            st.write(f"Using historical proxy rows for week {week}.")
            for pest_name, pack in models.items():
                df = pest_dfs.get(pest_name)
                if df is None:
                    st.warning(f"No CSV loaded for {pest_name}; cannot build historical proxy row. Skipping.")
                    continue
                # normalize columns
                df.columns = [c.replace("\n"," ").strip() for c in df.columns]
                # try exact week match, else modulo match, else last row
                if 'Standard Week' in df.columns and (df['Standard Week'] == week).any():
                    row = df[df['Standard Week']==week].iloc[-1]
                elif 'Standard Week' in df.columns and ((df['Standard Week'] % 52) == (week % 52)).any():
                    row = df[(df['Standard Week'] % 52) == (week % 52)].iloc[-1]
                else:
                    row = df.iloc[-1]
                Xrow = build_features_from_row(row, pack['features'], df_full=df)
                pred, risk = predict_from_pack(pack, Xrow)
                results.append({'Pest': pest_name, 'Predicted Count': int(round(pred,1)), 'Risk': risk, 'Year': int(row.get('Observation Year')) if not pd.isna(row.get('Observation Year')) else None })
            resdf = pd.DataFrame(results).sort_values(['Risk','Predicted Count'], ascending=[False, False])
            st.table(resdf)

        else:
            if not uploaded:
                st.warning("Upload a weekly CSV file (single row) to use this mode.")
            else:
                try:
                    wdf = pd.read_csv(uploaded)
                    wdf.columns = [c.replace("\n"," ").strip() for c in wdf.columns]
                    row = wdf.iloc[0]
                    st.write("Using uploaded weekly row for prediction.")
                    for pest_name, pack in models.items():
                        Xrow = build_features_from_row(row, pack['features'], df_full=wdf)
                        pred, risk = predict_from_pack(pack, Xrow)
                        results.append({'Pest': pest_name, 'Predicted Count': round(pred,1), 'Risk': risk})
                    resdf = pd.DataFrame(results).sort_values(['Risk','Predicted Count'], ascending=[False, False])
                    st.table(resdf)
                except Exception as e:
                    st.error(f"Failed to read uploaded file: {e}")
