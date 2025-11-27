"""
pest_prediction_pipeline.py

Single-file pipeline that trains per-pest RandomForest models using the improvements we discussed
(log1p target transform, pest lags, 2-week weather averages, time-based split) and exposes a
simple command-line "command center" for making weekly predictions.

Usage (examples):

# Train models for all CSVs in a folder (one CSV per pest):
python pest_prediction_pipeline.py train --data_dir ./data --models_dir ./models_all --min_rows 50

# Run CLI-style prediction for a week using historical rows to build features:
python pest_prediction_pipeline.py predict --models_dir ./models_all --data_dir ./data --week 10

# Predict for a single pest:
python pest_prediction_pipeline.py predict --models_dir ./models_all --data_dir ./data --week 10 --pest "Brownplanthopper"

# Predict using a user-supplied weekly input (if you have aggregated daily->weekly outside):
python pest_prediction_pipeline.py predict --models_dir ./models_all --weekly_input ./my_weekly.csv

Notes:
- Input CSVs are expected to be similar to your original files (they may have column headers
  with newlines like "Observation\nYear"; the script will normalize those to single-line names).
- The script looks for a column containing the word "Pest" (case-insensitive) and renames it to
  "PestValue" for internal use.
- Models are saved as joblib packages containing: model, features, imputer, transform, thresholds, metrics.

"""

from __future__ import annotations
import argparse
import os
import glob
import math
import json
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import math


# ------------------ Utility functions ------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace newlines and trim column names."""
    df = df.copy()
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    return df


def find_pest_column(columns: List[str]) -> Optional[str]:
    for c in columns:
        if 'pest' in c.lower():
            return c
    return None


# ------------------ Preprocessing ------------------

def load_and_preprocess(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load a pest CSV and compute features used for modelling.

    Returns (df, weather_columns). df contains original columns plus engineered features:
    - pest_lag1, pest_lag2
    - Prev2_<weather> for each weather column
    - week_sin, week_cos

    The function does not remove rows; caller may drop rows lacking lag values.
    """
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    pest_col = find_pest_column(list(df.columns))
    if pest_col is None:
        raise ValueError(f"No pest column detected in {csv_path}")
    # canonical name
    df = df.rename(columns={pest_col: 'PestValue'})

    # numeric conversion: keep 'Observation Year' and 'Standard Week' numeric if present
    for c in df.columns:
        if c in ['Observation Year', 'Standard Week']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            # strip non-numeric characters (degree symbols, % signs etc.) then convert
            df[c] = pd.to_numeric(df[c].astype(str).str.replace('[^0-9.+-eE]', '', regex=True), errors='coerce')

    # sort by Year+Week if available
    if 'Observation Year' in df.columns and 'Standard Week' in df.columns:
        df = df.sort_values(['Observation Year', 'Standard Week']).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # detect weather columns heuristically
    weather_cols = [c for c in df.columns if any(k in c for k in ['MaxT', 'MinT', 'RH', 'RF', 'WS', 'SSH', 'EVP', 'Temp', 'Humid', 'Rain'])]

    # interpolate weather columns to fill small gaps
    if len(weather_cols) > 0:
        df[weather_cols] = df[weather_cols].interpolate(limit_direction='both')

    # pest lags
    df['pest_lag1'] = df['PestValue'].shift(1)
    df['pest_lag2'] = df['PestValue'].shift(2)

    # Prev2 weather features: rolling mean of last two weeks, shifted by 1
    for c in weather_cols:
        df[f'Prev2_{c}'] = df[c].rolling(window=2, min_periods=1).mean().shift(1)

    # seasonality
    if 'Standard Week' in df.columns:
        df['stdweek'] = df['Standard Week'] % 52
        df['week_sin'] = np.sin(2 * np.pi * df['stdweek'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['stdweek'] / 52)
    else:
        df['week_sin'] = 0.0
        df['week_cos'] = 0.0

    return df, weather_cols


# ------------------ Training ------------------

def train_models_for_folder(data_dir: str, models_dir: str, min_rows: int = 50) -> List[Dict]:
    """Train a model for each CSV in data_dir and save into models_dir.

    Returns a list of summary records for each pest trained.
    """
    os.makedirs(models_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    summary = []

    for csv_path in csv_files:
        pest_basename = os.path.basename(csv_path)
        pest_name = pest_basename.replace('Rice_Maruteru - ', '').replace('(Number_ Light trap).csv', '').strip()
        try:
            df, weather_cols = load_and_preprocess(csv_path)
        except Exception as e:
            summary.append({'pest': pest_name, 'status': 'error', 'detail': str(e)})
            continue

        lag_feats = ['pest_lag1', 'pest_lag2']
        prev_feats = [f'Prev2_{c}' for c in weather_cols]
        feat_cols = [f for f in (lag_feats + ['week_sin', 'week_cos'] + prev_feats) if f in df.columns]

        # require lags available
        df_model = df.dropna(subset=['pest_lag1', 'pest_lag2']).copy()
        if len(prev_feats) > 0:
            # require at least one prev_feat present in columns
            required_prev = [c for c in prev_feats if c in df_model.columns]
            if required_prev:
                df_model = df_model.dropna(subset=required_prev, how='any')

        n_rows = len(df_model)
        if n_rows < min_rows:
            summary.append({'pest': pest_name, 'status': 'skipped', 'detail': f'not enough rows after lagging ({n_rows})'})
            continue

        used_feats = [c for c in feat_cols if c in df_model.columns]
        X_raw = df_model[used_feats].copy()
        imp = SimpleImputer(strategy='median')
        X = pd.DataFrame(imp.fit_transform(X_raw), columns=used_feats)

        y = df_model['PestValue'].fillna(0).astype(float)
        y_log = np.log1p(y)

        # time-based split
        n = len(X)
        train_idx = int(n * 0.8)
        if train_idx <= 10 or n - train_idx <= 5:
            summary.append({'pest': pest_name, 'status': 'skipped', 'detail': f'too few rows for time split ({n})'})
            continue

        X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:]
        y_train_log, y_test_log = y_log.iloc[:train_idx], y_log.iloc[train_idx:]

        # train model on log1p target
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train_log)

        # predict on test
        y_pred_log = rf.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test = np.expm1(y_test_log)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        baseline = float(np.median(np.expm1(y_train_log)))
        baseline_mae = mean_absolute_error(y_test, [baseline] * len(y_test))
        baseline_rmse = math.sqrt(mean_squared_error(y_test, np.full_like(y_test, baseline)))

        # thresholds from training original counts
        train_counts = np.expm1(y_train_log)
        p50, p75, p90 = float(np.percentile(train_counts, 50)), float(np.percentile(train_counts, 75)), float(np.percentile(train_counts, 90))

        model_pack = {
            'pest_name': pest_name,
            'model': rf,
            'features': used_feats,
            'imputer': imp,
            'transform': 'log1p',
            'thresholds': {'p50': p50, 'p75': p75, 'p90': p90},
            'metrics': {'mae': mae, 'rmse': rmse, 'baseline_mae': baseline_mae, 'baseline_rmse': baseline_rmse, 'train_rows': len(X_train), 'test_rows': len(X_test)}
        }

        outpath = os.path.join(models_dir, f"{pest_name.lower().replace(' ', '_')}_model.pkl")
        joblib.dump(model_pack, outpath)

        summary.append({'pest': pest_name, 'status': 'ok', 'mae': mae, 'rmse': rmse, 'baseline_mae': baseline_mae, 'baseline_rmse': baseline_rmse, 'p50': p50, 'p75': p75, 'p90': p90, 'model_path': outpath})

    return summary


# ------------------ Prediction / Command center ------------------

def build_features_for_prediction(row: pd.Series, features: List[str], df_full: pd.DataFrame) -> pd.DataFrame:
    """Given a historical row (or a row built from user weekly input), construct the DataFrame of features
    required by the model and impute missing values using medians from df_full when needed.
    """
    feat_vals = {}
    for f in features:
        if f in row.index:
            val = row.get(f, np.nan)
        else:
            val = np.nan
        # if lag is missing, try median
        if (f in ['pest_lag1', 'pest_lag2']) and (pd.isna(val) or val is None):
            if 'PestValue' in df_full.columns:
                val = float(df_full['PestValue'].median())
        feat_vals[f] = val

    Xrow = pd.DataFrame([feat_vals], columns=features)
    return Xrow


def predict_from_saved_model(model_pack: Dict, Xrow: pd.DataFrame) -> Tuple[float, str]:
    """Return predicted count (original scale) and risk label using thresholds inside model_pack."""
    imp = model_pack.get('imputer')
    if imp is not None:
        X_imp = pd.DataFrame(imp.transform(Xrow), columns=model_pack['features'])
    else:
        X_imp = Xrow.fillna(0)

    model = model_pack['model']
    if model_pack.get('transform') == 'log1p':
        pred_log = model.predict(X_imp)[0]
        pred = float(np.expm1(pred_log))
    else:
        pred = float(model.predict(X_imp)[0])

    thr = model_pack.get('thresholds', None)
    if thr is None:
        # fallback thresholds
        if pred <= 0:
            risk = 'None'
        elif pred <= 2:
            risk = 'Low'
        elif pred <= 10:
            risk = 'Medium'
        else:
            risk = 'High'
    else:
        if pred <= 0:
            risk = 'None'
        elif pred <= thr['p50']:
            risk = 'Low'
        elif pred <= thr['p75']:
            risk = 'Medium'
        else:
            risk = 'High'

    return pred, risk


# ------------------ CLI glue ------------------

def cmd_train(args):
    summary = train_models_for_folder(args.data_dir, args.models_dir, min_rows=args.min_rows)
    print('\nTraining summary (first 10 items):')
    for s in summary[:20]:
        print(json.dumps(s, indent=2, default=str))
    # save overall summary
    summary_path = os.path.join(args.models_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary to {summary_path}")


def cmd_predict(args):
    # load models
    model_files = sorted(glob.glob(os.path.join(args.models_dir, '*.pkl')))
    if not model_files:
        raise SystemExit('No models found in models_dir')
    models = {}
    for mf in model_files:
        pack = joblib.load(mf)
        models[pack['pest_name']] = pack

    # load data_dir CSVs to source historical rows (if provided)
    pest_dfs = {}
    if args.data_dir:
        csv_files = sorted(glob.glob(os.path.join(args.data_dir, '*.csv')))
        for csv_path in csv_files:
            pest = os.path.basename(csv_path).replace('Rice_Maruteru - ', '').replace('(Number_ Light trap).csv', '').strip()
            try:
                df, _ = load_and_preprocess(csv_path)
                pest_dfs[pest] = df
            except Exception as e:
                print('Warning: failed to load', csv_path, e)

    # if weekly_input is provided, use that single-row (must contain weekly-aggregated fields)
    if args.weekly_input:
        df_weekly = pd.read_csv(args.weekly_input)
        df_weekly = normalize_columns(df_weekly)
        # assume single-row
        row = df_weekly.iloc[0]

        results = []
        for pest_name, pack in models.items():
            # build features from row; no df_full available
            Xrow = build_features_for_prediction(row, pack['features'], df_weekly)
            pred, risk = predict_from_saved_model(pack, Xrow)
            results.append({'pest': pest_name, 'predicted_count': pred, 'risk': risk})

        print('Results from weekly_input:')
        for r in results:
            print(r)
        return

    # Otherwise use historical rows per pest and the requested week
    if args.week is None:
        raise SystemExit('Please pass --week N when using historical mode')
    week = int(args.week)

    pests_to_run = [args.pest] if args.pest else list(models.keys())
    out_rows = []
    for pest_name in pests_to_run:
        if pest_name not in models:
            print('No model for', pest_name)
            continue
        model_pack = models[pest_name]
        df_full = pest_dfs.get(pest_name)
        if df_full is None:
            print(f'No historical CSV loaded for {pest_name}; skipping')
            continue
        # select a row for that week (most recent year)
        if 'Standard Week' in df_full.columns and (df_full['Standard Week'] == week).any():
            row = df_full[df_full['Standard Week'] == week].iloc[-1]
        else:
            # fallback on modulo match
            if 'Standard Week' in df_full.columns and ((df_full['Standard Week'] % 52) == (week % 52)).any():
                row = df_full[(df_full['Standard Week'] % 52) == (week % 52)].iloc[-1]
            else:
                row = df_full.iloc[-1]

        Xrow = build_features_for_prediction(row, model_pack['features'], df_full)
        pred, risk = predict_from_saved_model(model_pack, Xrow)
        out_rows.append({'pest': pest_name, 'week': week, 'predicted_count': pred, 'risk': risk, 'year': row.get('Observation Year', None)})

    # print table
    df_out = pd.DataFrame(out_rows)
    print('\nPrediction results:')
    print(df_out.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Pest prediction pipeline (train + simple command center)')
    sub = parser.add_subparsers(dest='cmd')

    # train
    p_train = sub.add_parser('train')
    p_train.add_argument('--data_dir', required=True, help='Directory with pest CSV files')
    p_train.add_argument('--models_dir', required=True, help='Directory to save trained models')
    p_train.add_argument('--min_rows', type=int, default=50, help='Minimum rows after lagging to train')

    # predict
    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--models_dir', required=True, help='Directory with trained models')
    p_pred.add_argument('--data_dir', help='Directory with historical CSVs (used to build features for weeks)')
    p_pred.add_argument('--week', type=int, help='Standard week number to simulate (1-52)')
    p_pred.add_argument('--pest', type=str, help='Optional: single pest name to predict (must match saved model pest_name)')
    p_pred.add_argument('--weekly_input', type=str, help='Optional: path to single-row weekly input CSV to use directly')

    args = parser.parse_args()
    if args.cmd == 'train':
        cmd_train(args)
    elif args.cmd == 'predict':
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
