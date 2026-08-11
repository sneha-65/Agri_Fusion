# 🌾 Agri Fusion — AI-Powered Smart Farming Platform

> **Turning agricultural data into simple, practical decisions for farmers.**

🔗 **Live Application:** [KrishiSense](https://krishisense.streamlit.app/)

📍 Designed for farmers across **Andhra Pradesh & Telangana**

---

## 🌱 Overview

**Agri Fusion** is an AI-powered smart agriculture platform designed to help farmers make better farming decisions using **Machine Learning, weather data, soil information, crop data, and market information**.

Instead of presenting complex agricultural data, the platform converts predictions into simple questions that matter to a farmer:

* 🌾 **What crop should I grow?**
* 🌡️ **Is today's climate safe for my crop?**
* 💧 **Should I irrigate today?**
* 📈 **How much harvest can I expect?**
* 💰 **What could my crop be worth?**

The goal is not to replace the farmer's experience, but to combine **farmer knowledge + agricultural science + data + AI** to support better decisions.

---

# 🎯 Problem Statement

Farmers make important decisions throughout every crop season, but these decisions are affected by constantly changing environmental and economic conditions.

### Major challenges

* 🌦️ Unpredictable weather and climate conditions
* 🌱 Choosing unsuitable crops for local conditions
* 💧 Inefficient irrigation and water wastage
* 🌾 Uncertainty about expected crop yield
* 💰 Uncertainty about agricultural market prices
* 📊 Difficulty interpreting large amounts of agricultural data

Agri Fusion brings these different factors together into a single farmer-oriented platform.

---

# 🚜 What Agri Fusion Provides

| AI Module              | Farmer's Question         | Output                                           |
| ---------------------- | ------------------------- | ------------------------------------------------ |
| 🌱 Crop Recommendation | What should I grow?       | Recommended crop + confidence                    |
| 🌡️ Climate Risk       | Is my crop at risk today? | Risk level + actions                             |
| 💧 Irrigation Advisor  | Should I water today?     | Irrigation decision + water requirement + timing |
| 📈 Yield Estimator     | How much can I harvest?   | Estimated crop yield                             |
| 💰 Market Price        | What price can I expect?  | Estimated market price                           |
| 🚜 Farm Plan           | What should I do today?   | Combined farming recommendations                 |

---

# 🌱 1. Crop Recommendation

The Crop Recommendation module uses agricultural and weather features to recommend a suitable crop.

### Inputs

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* Soil pH
* Rainfall

The farmer only needs to select the **city** in the simplified interface. Weather and regional agricultural information are automatically retrieved.

### Output

Example:

> 🌱 **Recommended Crop: Groundnut**
> **AI Confidence: 96%**

The application also explains **why the crop was recommended** instead of showing only the prediction.

### Machine Learning

**Problem:** Multi-Class Classification

**Target:** Crop label

**Models:** Multiple classification models were evaluated and the best-performing model was selected.

---

# 🌡️ 2. Climate Risk Early Warning

The Climate Risk module analyzes current and historical weather conditions to identify potential agricultural climate risks.

### Factors considered

* 🌡️ Temperature
* 💧 Humidity
* 🌧️ Rainfall
* 💨 Wind
* ☀️ Solar radiation
* 💦 ET₀
* 🌧️ Recent rainfall history
* 🔥 Heat conditions
* 🌱 Soil conditions
* 📅 Dry-spell information

### Risk Categories

🟢 **Low**

🟡 **Moderate**

🟠 **High**

🔴 **Extreme**

The system also provides practical recommendations such as continuing normal monitoring or taking additional precautions.

### Model

**Problem:** Multi-Class Classification

**Target:** Climate Risk

**Classes:** Low / Moderate / High / Extreme

**Model:** Random Forest

**Test Accuracy:** ~86.6%

---

# 💧 3. Irrigation Advisor

The Irrigation Advisor is designed around the farmer's most important question:

> **"Should I water my crop today?"**

Instead of showing only technical values such as `mm/day`, the system converts the prediction into practical irrigation guidance.

### The system considers

* 🌾 Crop
* 🌱 Growth stage
* 📍 Location
* 🚜 Farm size
* 🌦️ Temperature
* 💧 Humidity
* 🌧️ Rainfall
* 💦 ET₀
* 💨 Wind
* ☀️ Solar radiation
* 💧 Irrigation method
* ⚙️ Pump flow rate

### Farmer-oriented output

The system can tell the farmer:

**💧 IRRIGATION REQUIRED TODAY**

or

**❌ NO IRRIGATION NEEDED TODAY**

It also provides:

* Estimated water requirement
* Best irrigation time
* Next recommended watering date
* Pump running time when pump flow is available
* Multi-day irrigation schedule

For example:

> **44,000 L approximately**
> Estimated water requirement for the farm

The application also explains that the value represents the **estimated total water required for the selected farm area**, rather than leaving the farmer to interpret `mm/day`.

### ML Problem

**Problem:** Regression

**Target:** Water requirement (mm/day)

**Training Records:** ~357,504

**Formula Basis:** FAO-56 crop water requirement methodology

**Best Model:** Linear Regression

**Test R²:** 0.9113

**Test RMSE:** 0.9258 mm/day

---

# 📈 4. Yield Estimator

The Yield Estimator predicts the expected crop yield based on farm, location, weather, soil and crop characteristics.

### Features include

* State
* District
* Latitude
* Longitude
* Elevation
* Farm area
* Year
* Crop
* Season
* Temperature
* Rainfall
* Humidity
* Solar radiation
* Wind speed
* ET₀
* Soil properties

### Example output

> 🌾 **Expected Harvest: 560 kg**
> For a **2-acre Groundnut farm**

The system also converts the result into **quintals** so that the farmer can understand the expected production more easily.

### Model

**Problem:** Regression

**Training Records:** 15,276

**Best Model:** Random Forest

**Test R²:** 0.9661

**Test RMSE:** 2.98 tonnes/ha

---

# 💰 5. Market Price Prediction

The Market Price module estimates the expected agricultural market price using historical market information.

### Features

* Commodity
* State
* District
* Arrival quantity
* Day
* Month
* Year
* Quarter

### Example

> 💰 **Estimated Market Price**
> ₹2,000 per quintal

The system can also calculate the approximate gross value of the predicted harvest.

### Model

**Problem:** Regression

**Target:** Market Price (₹/quintal)

**Model:** Tuned Random Forest Regressor

**Test R²:** 0.9603

**MAE:** ₹189.78

**RMSE:** ₹534.38

---

# 🚜 6. Farm Plan

The **Farm Plan** combines multiple predictions into one simplified farmer dashboard.

Instead of making the farmer visit five different modules, it answers:

### 🌱 What should I plant?

Recommended crop based on agricultural conditions.

### 🌦️ Is today suitable?

Current weather and climate conditions.

### 💧 Should I irrigate?

Water requirement and irrigation recommendation.

### 🌾 How much could I harvest?

Expected crop production.

### 💰 What could it be worth?

Estimated market value.

The purpose is to provide a **simple daily decision-support system** rather than overwhelming the farmer with raw model outputs.

---

# 🌐 Data Sources

Agri Fusion combines multiple agricultural and meteorological data sources.

### 🌤️ Open-Meteo

Used for live and historical weather information.

Data includes:

* Temperature
* Humidity
* Rainfall
* Wind
* Solar radiation
* ET₀
* Other weather variables

🔗 https://open-meteo.com/

---

### 🧪 SoilGrids — ISRIC

Used for soil-related information such as:

* Soil pH
* Organic carbon
* Sand
* Silt
* Clay
* CEC
* Bulk density
* Water-related soil properties

🔗 https://www.isric.org/explore/soilgrids

---

### 📚 FAO-56

Used as the agricultural/scientific basis for crop water requirement calculations, including crop coefficients and evapotranspiration concepts.

🔗 https://www.fao.org/

---

### 📊 Government Agricultural Data

Agricultural yield and production information was sourced from publicly available Indian agricultural datasets.

🔗 https://data.gov.in/

---

### 💰 Agricultural Market Data

Market-price information was obtained from publicly available agricultural market datasets including Agmarknet-related data.

---

# 🤖 Machine Learning Architecture

Agri Fusion uses different models for different agricultural problems rather than attempting to solve every problem with one model.

```text
                    FARMER INPUT
                         │
                         ▼
                ┌─────────────────┐
                │  Agri Fusion UI │
                │    Streamlit     │
                └────────┬────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
       Weather          Soil          Farmer
        Data            Data           Data
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  ML Prediction  │
                │     Layer       │
                └────────┬────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Crop Model       Climate Model    Irrigation Model
        │                │                │
        └────────────────┼────────────────┘
                         │
                 ┌───────┴────────┐
                 ▼                ▼
            Yield Model      Market Model
                 │                │
                 └───────┬────────┘
                         ▼
                 FARMER-FRIENDLY
                    DECISION
```

---

# 🛠️ Technology Stack

### Programming

* Python

### Machine Learning

* Scikit-learn
* Random Forest
* Linear Regression
* XGBoost
* Classification & Regression techniques

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Streamlit visual components

### Application

* Streamlit
* Python backend modules

### APIs

* Open-Meteo API
* Supabase

### Model Storage

* Pickle / Joblib

### Development

* Jupyter Notebook
* Git
* GitHub

---

# 🖥️ Application Architecture

```text
                    ┌────────────────────┐
                    │      Farmer        │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Streamlit Frontend │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        Crop Module     Climate Module   Irrigation Module
              │               │               │
              └───────────────┼───────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Yield Module     Market Module     Supabase
              │               │
              └───────────────┼───────────────┘
                              ▼
                     Farmer Dashboard
```

---

# 📊 Model Performance

| Module                 | Problem        | Best Model                | Performance    |
| ---------------------- | -------------- | ------------------------- | -------------- |
| 🌱 Crop Recommendation | Classification | Best evaluated classifier | —              |
| 🌡️ Climate Risk       | Classification | Random Forest             | 86.6% Accuracy |
| 💧 Irrigation          | Regression     | Linear Regression         | R² 91.13%      |
| 📈 Yield               | Regression     | Random Forest             | R² 96.61%      |
| 💰 Market Price        | Regression     | Tuned Random Forest       | R² 96.03%      |

> Model performance is based on the project's test datasets and should not be interpreted as a guarantee of real-world prediction accuracy.

---

# 📁 Project Structure

```text
Agri_Fusion/
│
├── App/
│   ├── frontend/
│   │   └── app1.py
│   │
│   ├── backend/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── predict.py
│   │   ├── weather.py
│   │   ├── climate_risk.py
│   │   ├── yield_estimator.py
│   │   ├── crop_recommendation.py
│   │   └── ...
│   │
│   ├── Pickles/
│   │   ├── Crop/
│   │   ├── Climate/
│   │   ├── Ir/
│   │   ├── Yield/
│   │   └── Market/
│   │
│   ├── requirements.txt
│   └── .env
│
├── Notebooks/
│   ├── Climate_Risk.ipynb
│   ├── Crop_Recommendation_NB.ipynb
│   ├── Irrigation_Model_NB.ipynb
│   ├── Market_prd.ipynb
│   └── yield-Copy1.ipynb
│
├── Data/
│
├── .gitignore
└── README.md
```

---

# 🚀 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/sneha-65/agri-fusion.git
cd agri-fusion
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r App/requirements.txt
```

### 5. Configure environment variables

Create:

```text
App/.env
```

Add your required Supabase/API configuration.

> Never commit `.env` or other secrets to GitHub.

### 6. Run the application

```bash
streamlit run App/frontend/app1.py
```

---

# ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Demo

🚀 **[Open Agri Fusion](https://krishisense.streamlit.app/)**

The application automatically retrieves live weather information through the configured weather API and uses the trained machine learning models to generate predictions.

---

# 👨‍🌾 Farmer-Centered Design

A major design principle of Agri Fusion is:

> **Complex AI should produce simple farming decisions.**

Instead of expecting farmers to understand:

```text
ET₀
Kc
mm/day
R²
RMSE
One-Hot Encoding
Feature Scaling
```

the application attempts to communicate:

```text
💧 Water today or not?

🌱 Which crop is suitable?

🌦️ Is today's weather risky?

🌾 How much harvest can I expect?

💰 What price might I receive?
```

Technical information remains available for reference, while the primary interface focuses on practical decisions.

---

# ⚠️ Disclaimer

Agri Fusion predictions are generated by Machine Learning models trained using historical agricultural, weather, soil and market data.

The predictions are **estimates and not guarantees**.

Actual farming outcomes depend on many factors including:

* Local weather
* Soil conditions
* Seed variety
* Irrigation availability
* Fertilizer application
* Pest and disease pressure
* Farm management
* Market conditions

Farmers should use the system as a **decision-support tool** and consult local agricultural experts for critical farming decisions.

---

# 🔮 Future Improvements

* 📱 Farmer-friendly mobile application
* 🗣️ Telugu voice-based agricultural assistant
* 🌧️ Improved short-term rainfall forecasting
* 🛰️ Satellite-based crop monitoring
* 🦠 Crop disease detection
* 📍 More district-level agricultural data
* 💧 Better farm-specific irrigation calibration
* 📊 Real-time market integration
* 🤖 Personalized AI farming assistant
* 🌾 IoT-based soil moisture integration

---

# 👩‍💻 Author

**Sneha Kammara**

B.Tech — Computer Science & Engineering

Interested in **Machine Learning, Data Science and AI-powered agricultural solutions**.

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

### 🌾 Agri Fusion

**AI + Agriculture + Weather + Soil + Market Data**

> **Better information → Better decisions → Smarter farming**
