# Car Price Prediction using Machine Learning

A machine learning project that predicts the **resale price of used cars** based on real-world ex-showroom prices. The project includes **138 real car models** across **17 brands** with actual showroom prices, and uses an **Ensemble ML model (Random Forest + Gradient Boosting)** to predict resale value with **99.2% R² accuracy**.

Users interact through a simple **Tkinter GUI** — select a brand, pick a model (showroom price auto-fills), enter car details, and get an instant resale price prediction.

---

## Project Overview

| Detail              | Description                                        |
| ------------------- | -------------------------------------------------- |
| **Project Title**   | Car Price Prediction using ML                      |
| **Domain**          | Machine Learning — Supervised Learning (Regression)|
| **Algorithm**       | Ensemble (Random Forest + Gradient Boosting)       |
| **Accuracy (R2)**   | 99.2%                                              |
| **Avg Error (MAE)** | +-Rs.0.81 Lakhs                                    |
| **Training Data**   | 10,000 records from 138 real car models            |
| **Features**        | 17 (12 base + 5 engineered)                        |
| **Car Brands**      | 17 (Maruti, Hyundai, Tata, BMW, Mercedes, etc.)    |
| **Car Models**      | 138 real models with actual ex-showroom prices     |
| **Language**        | Python 3.x                                         |
| **UI**              | Tkinter (Desktop GUI)                              |

---

## Objective

Predict the resale/selling price of a used car based on:
- Real ex-showroom price of the car model
- Car age, kilometers driven, fuel type, transmission
- Number of previous owners, seller type, mileage, engine CC

The user selects a brand and model from real car data, and the ML ensemble predicts the current resale value.

---

## Project Structure

```
Car-Price-Prediction/
|
|-- car_price_prediction.py    # Main file — ML model + Tkinter UI
|-- car_dataset.csv            # Generated training dataset
|-- README.md                  # Project documentation (this file)
|
|-- model/
|   +-- car_price_model.pkl    # Saved trained model (joblib)
|
+-- plots/
    |-- 01_target_distribution.png
    |-- 02_correlation_heatmap.png
    |-- ... (10 visualization charts)
    +-- 10_feature_importance.png
```

---

## Real Car Database (138 Models, 17 Brands)

The model uses actual ex-showroom prices from the Indian market (2024-2025):

| Brand      | Models | Price Range (Lakhs)  | Example Models                          |
| ---------- | ------ | -------------------- | --------------------------------------- |
| Maruti     | 17     | Rs.3.99 - Rs.24.79   | Alto K10, Swift, Baleno, Brezza, Grand Vitara |
| Hyundai    | 12     | Rs.5.92 - Rs.44.95   | i20, Venue, Creta, Verna, Alcazar, Ioniq 5 |
| Tata       | 12     | Rs.5.65 - Rs.16.19   | Punch, Nexon, Harrier, Safari, Nexon EV |
| Mahindra   | 11     | Rs.7.49 - Rs.15.49   | Thar, Scorpio N, XUV700, XUV400 EV     |
| Toyota     | 10     | Rs.6.86 - Rs.210.00  | Glanza, Innova, Fortuner, Camry         |
| Kia        | 6      | Rs.7.99 - Rs.60.95   | Sonet, Seltos, Carens, EV6              |
| Honda      | 5      | Rs.7.20 - Rs.19.50   | Amaze, City, Elevate                    |
| BMW        | 12     | Rs.38.00 - Rs.195.00 | 3 Series, X1, X5, i4, M340i            |
| Mercedes   | 14     | Rs.45.00 - Rs.162.00 | C-Class, E-Class, GLA, GLE, EQS        |
| Audi       | 10     | Rs.43.81 - Rs.180.00 | A4, Q5, Q7, e-tron GT                   |
| MG         | 7      | Rs.7.98 - Rs.38.80   | Hector, Astor, ZS EV, Gloster          |
| Volkswagen | 4      | Rs.7.98 - Rs.35.17   | Polo, Virtus, Taigun, Tiguan           |
| Skoda      | 4      | Rs.10.69 - Rs.54.00  | Slavia, Kushaq, Superb, Kodiaq         |
| Renault    | 3      | Rs.4.70 - Rs.6.50    | Kwid, Triber, Kiger                     |
| Nissan     | 2      | Rs.6.00 - Rs.49.92   | Magnite, X-Trail                        |
| Jeep       | 4      | Rs.18.99 - Rs.77.50  | Compass, Wrangler, Grand Cherokee       |
| Citroen    | 5      | Rs.6.16 - Rs.36.91   | C3, C3 Aircross, Basalt                |

---

## ML Concepts Used

### 1. Dataset Generation
- 10,000 used-car records generated from 138 real car models
- Realistic depreciation model applied to actual ex-showroom prices
- Depreciation factors: Year 1 = 15-20%, Year 2 = 10%, then 7% per year
- Additional factors: km driven, owners, fuel type, transmission, seller type
- Gaussian noise (6% variance) for real-world simulation

### 2. Data Preprocessing
- **Label Encoding** — Brand, Model, Fuel Type, Transmission, Seller Type encoded using `LabelEncoder`
- **Feature Engineering** — 5 derived features:
  - `Km_per_Year` — Usage intensity (km driven / car age)
  - `Age_Squared` — Non-linear depreciation capture
  - `Price_per_Age` — Showroom price divided by age (value retention rate)
  - `Power_Weight` — Engine CC per seat (power-to-size ratio)
  - `Log_Km` — Log-transformed km driven (reduces skewness)
- **Feature Scaling** — StandardScaler applied to all 17 features
- **Train-Test Split** — 80% training / 20% testing

### 3. Model Training & Results

| Model                              | R2 Score | Notes                    |
| ---------------------------------- | -------- | ------------------------ |
| Random Forest (300 trees)          | 98.94%   | Strong baseline          |
| Gradient Boosting (300 trees)      | 99.20%   | Best individual model    |
| **Ensemble (GBR x0.85 + RF x0.15)** | **99.20%** | **Final model used** |

### 4. Evaluation
- **R2 Score**: 99.2% — model explains 99.2% of price variance
- **MAE**: +-Rs.0.81 Lakhs average error
- **5-Fold Cross Validation R2**: 98.96% — confirms no overfitting
- **Optimal ensemble weight** found via grid search (GBR=0.85, RF=0.15)

### 5. Key Feature: Real Showroom Prices
Unlike synthetic-only datasets, this model uses **actual ex-showroom prices** as a feature. This means:
- A BMW 3 Series (Rs.47.90L showroom) will predict differently than a Maruti Swift (Rs.6.49L)
- The model learns real depreciation patterns per price segment
- Predictions are grounded in actual market values

---

## Features Used for Prediction (17 Total)

### Base Features (12)
| Feature        | Type        | Description                          |
| -------------- | ----------- | ------------------------------------ |
| Brand          | Categorical | Car manufacturer (17 brands)         |
| Model          | Categorical | Car model name (138 models)          |
| Showroom_Price | Numerical   | Real ex-showroom price in Lakhs      |
| Fuel Type      | Categorical | Petrol, Diesel, CNG, Electric        |
| Transmission   | Categorical | Manual, Automatic                    |
| Seller Type    | Categorical | Dealer, Individual, Trustmark Dealer |
| Km Driven      | Numerical   | Total kilometers driven              |
| Mileage        | Numerical   | Fuel efficiency (km/l)               |
| Engine CC      | Numerical   | Engine displacement                  |
| Seats          | Numerical   | Number of seats (4-8)                |
| Owners         | Numerical   | Previous owners (1-5)                |
| Car Age        | Numerical   | 2025 - Manufacturing Year            |

### Engineered Features (5)
| Feature       | Description                                      |
| ------------- | ------------------------------------------------ |
| Km_per_Year   | Km Driven / Car Age (usage intensity)            |
| Age_Squared   | Car Age squared (non-linear depreciation)        |
| Price_per_Age | Showroom Price / (Car Age + 1) (value retention) |
| Power_Weight  | Engine CC / Seats (power-to-size ratio)          |
| Log_Km        | log(1 + Km Driven) (reduces skewness)            |

---

## How to Run

### Prerequisites
- Python 3.8 or higher
- Required packages:
```bash
pip install numpy pandas scikit-learn
```

### Run the Application
```bash
python car_price_prediction.py
```

### What Happens
1. Model trains on 10,000 records from real car data (takes ~10 seconds)
2. GUI window opens with a dark theme
3. Select **Brand** -> **Model** dropdown auto-updates, showroom price auto-fills
4. Enter Year, Km Driven, Mileage, Engine CC, Owners
5. Click **"PREDICT RESALE PRICE"**
6. See: Estimated resale price, price range, depreciation %, and market segment

---

## UI Features

- **Brand -> Model linking**: Selecting a brand auto-populates the model dropdown with that brand's cars
- **Auto showroom price**: Selecting a model auto-fills the real ex-showroom price
- **Depreciation display**: Shows how much value the car has lost from showroom price
- **Market segments**: Entry-Level (<5L), Mid-Range (5-15L), Premium (15-35L), Luxury (35-60L), Ultra Luxury (60L+)
- **Input validation**: Warns if values are out of range

---

## ML Pipeline

```
+----------------------------+
| 1. Real Car Database       |  138 models, 17 brands, actual showroom prices
+------------+---------------+
             v
+----------------------------+
| 2. Generate Training Data  |  10,000 records with realistic depreciation
+------------+---------------+
             v
+----------------------------+
| 3. Feature Engineering     |  5 derived features (Km_per_Year, Age2, etc.)
+------------+---------------+
             v
+----------------------------+
| 4. Preprocessing           |  Label Encoding -> Standard Scaling -> 80/20 Split
+------------+---------------+
             v
+----------------------------+
| 5. Train Ensemble          |  Random Forest (300) + Gradient Boosting (300)
+------------+---------------+
             v
+----------------------------+
| 6. Optimal Blending        |  GBR x0.85 + RF x0.15 = 99.2% R2
+------------+---------------+
             v
+----------------------------+
| 7. Tkinter UI              |  Brand -> Model -> Auto showroom price -> Predict
+----------------------------+
```

---

## Tech Stack

| Component       | Technology                                |
| --------------- | ----------------------------------------- |
| Language        | Python 3.x                                |
| ML Library      | Scikit-Learn                              |
| Models          | Random Forest + Gradient Boosting (Ensemble) |
| Data Handling   | Pandas, NumPy                             |
| Visualization   | Matplotlib, Seaborn                       |
| GUI             | Tkinter                                   |
| Model Saving    | Joblib                                    |

---

## Future Improvements

- Connect to a live car pricing API for real-time showroom prices
- Add car images in the UI for each model
- Deploy as a web app using Flask or Streamlit
- Add more car brands (Porsche, Volvo, Land Rover, etc.)
- Include car variant-level pricing (base, mid, top)
- Add historical price trend charts

---

## Contributors

- Car Price Prediction Team

---

## License

This project is for educational purposes. Feel free to use and modify.
