# 🌾 Pest Prediction for Rice (Maruteru) — Machine Learning Prototype

### Forecasting weekly pest pressure using historical light-trap data and weather attributes

This repository contains a working machine learning prototype built to **predict pest outbreaks in rice crops** for the **Maruteru region (Andhra Pradesh)**.  
The system learns pest population behavior from historical trap counts and corresponding weekly weather conditions, and predicts **expected pest count + risk level** for a future week.

---

## 🎯 Objective

The goal of this prototype is to:

- Understand the relationship between weather and pest growth
- Predict weekly pest pressure ahead of time
- Provide interpretable risk levels to support pest management decisions
- Establish a reusable pipeline for scaling to multiple pests & locations

This is an **initial working model**, intended for demonstration and discussion with domain experts.

---

## 🐛 Supported Pests (Rice – Maruteru)

Each pest has its own dataset and its own machine learning model:

| Pest |
|------|
| Brown Planthopper |
| Whitebacked Planthopper |
| Green Leafhopper |
| Leaf Folder |
| ZigZag Leafhopper |
| Yellow Stem Borer |
| Gall Midge |
| Mirid Bug |
| Caseworm |

---

## 🧠 Methodology Overview

### 📌 Input Data Sources
Each pest dataset includes:

- **Observation Year**
- **Standard Week**
- **Light-trap count**
- **Weather parameters:**
  - Max/Min Temperature
  - Relative Humidity (RH1, RH2)
  - Rainfall (RF)
  - Wind Speed (WS)
  - Sunshine Hours (SSH)
  - Evaporation (EVP)

### 🧮 Feature Engineering

| Feature Type | Description |
|--------------|-------------|
| Pest History | Previous weeks’ counts (`pest_lag1`, `pest_lag2`) |
| Weather Aggregates | 2-week average exposure features |
| Seasonal Encoding | Week encoded using sinusoidal seasonality |

### 🤖 Model Details

- **Algorithm:** RandomForestRegressor  
- **Target Transformation:** `log1p(PestValue)` to handle extreme outbreak spikes  
- **Inverse Prediction:** `expm1(pred)` to return real counts  
- **Validation:** Time-based train/test split (no leakage across future weeks)
- **Risk Levels:** Derived from training percentiles of actual pest counts  
  - Low (≤ p50), Medium (p50–p75), High (≥ p75), None (≤ 0)

---

## 🔧 Training the Models

Train all per-pest models using:

```bash
python pest_prediction_pipeline.py train --data_dir ./data --models_dir ./models_all --min_rows 50
```
### 📤 Output Artifacts
This produces:
- ✔ **Per-pest trained model files** (`.pkl`) in `models_all/`
- ✔ `training_summary.json` with model metrics and thresholds

### 🔮 Predicting (Command Line)

**Predict for all pests for a selected Standard Week:**

```bash
python pest_prediction_pipeline.py predict --models_dir ./models_all --data_dir ./data --week 10
```
### Predict only one specific pest:
```bash
python pest_prediction_pipeline.py predict --models_dir ./models_all --data_dir ./data --week 10 --pest "Brownplanthopper"
```
### Predict using a custom single-week CSV input:
```bash
python pest_prediction_pipeline.py predict --weekly_input sample.csv
```
# 🌐 Web Demo (Streamlit UI)

## 🚀 Launch Instructions
Launch the prototype app using the following command:

```bash
streamlit run app.py
```
# Current UI Features

* **Standard Week Selection:** Choose a Standard Week to generate predictions using historical weather as a proxy.
* **Custom Data Testing:** Upload a single-row weekly CSV to test real data.
* **Readability:** Pest counts are displayed as integers for clear interpretation.

> **💡 Note:** This is a prototype UI and does not yet include weather visuals, color-coded badges, or CSV export. These will be added in upcoming iterations.

---

# 📍 Known Limitations & Next Milestones

## 🚧 Current Limitations
* **Weather Data:** Weather for predictions is currently sampled from historical weeks, not real forecasts.
* **Risk Thresholds:** Thresholds are automatically derived and have not yet been validated by agronomy experts.
* **Uncertainty:** Uncertainty ranges (confidence intervals) are not yet provided.

## 📌 Planned Enhancements

| Area | Enhancement |
| :--- | :--- |
| **Weather** | Integrate real forecast API + weekly aggregator |
| **UX** | Add risk badges, weather panel, and CSV export |
| **Modeling** | Add uncertainty estimation + model explainability |
| **Deployment** | REST API + scalable multi-location support |
| **Domain** | Expert-validated risk thresholds and actionable advisories |

---

# 🏁 Conclusion

This prototype demonstrates that weather-driven pest forecasting is feasible and can meaningfully assist pest management.

With expert validation and a real weather feed, the system can evolve into a scalable decision support tool for sustainable agriculture.

---

# ✉ Contact

**Sanidhya Kumar Ghosal**
*Project Associate — Annam.AI*
*B.Tech (IT)*

🌱 **Open to collaboration & research discussion!**
