# 🚗 Car Price Prediction using Machine Learning

A machine learning project that predicts the selling price of used cars based on various features like brand, fuel type, transmission, mileage, engine capacity, and more. The project includes a simple **Tkinter GUI** where users can enter car details and get an instant price prediction powered by a trained **Gradient Boosting Regressor** model.

---

## 📌 Project Overview

| Detail            | Description                                      |
| ----------------- | ------------------------------------------------ |
| **Project Title** | Car Price Prediction using ML                    |
| **Domain**        | Machine Learning — Supervised Learning (Regression) |
| **Algorithm**     | Ensemble (Random Forest + Gradient Boosting)     |
| **Accuracy (R²)** | ~96%                                            |
| **Avg Error (MAE)** | ±₹0.11 Lakhs                                 |
| **Training Data** | 10,000 synthetic samples                         |
| **Features**      | 15 (10 base + 5 engineered)                      |
| **Language**      | Python 3.x                                       |
| **UI**            | Tkinter (Desktop GUI)                            |

---

## 🎯 Objective

Build a machine learning model that can accurately predict the resale/selling price of a used car based on user-provided features. The user enters car details through a simple desktop UI, and the trained ML ensemble returns the estimated price in Indian Rupees (Lakhs).

---

## 📂 Project Structure

```
Car-Price-Prediction/
│
├── car_price_prediction.py    # Main file — ML model + Tkinter UI
├── car_dataset.csv            # Generated training dataset (5000 samples)
├── README.md                  # Project documentation (this file)
│
├── model/
│   └── car_price_model.pkl    # Saved trained model (joblib)
│
└── plots/
    ├── 01_target_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_price_by_brand.png
    ├── 04_price_vs_km.png
    ├── 05_price_vs_year.png
    ├── 06_categorical_counts.png
    ├── 07_model_r2_comparison.png
    ├── 08_mae_rmse_comparison.png
    ├── 09_actual_vs_predicted.png
    └── 10_feature_importance.png
```

---

## 🧠 Machine Learning Concepts Used

### 1. Dataset Generation
- 10,000 synthetic car records generated using NumPy with realistic distributions
- Features include brand premiums, depreciation curves, fuel-type bonuses, and Gaussian noise to simulate real-world variance
- Reduced noise (σ=0.25) compared to earlier version for cleaner signal

### 2. Data Preprocessing
- **Label Encoding** — Categorical features (Brand, Fuel Type, Transmission, Seller Type) converted to numerical values using `sklearn.preprocessing.LabelEncoder`
- **Feature Engineering** — 5 new derived features created:
  - `Km_per_Year` — Average km driven per year (usage intensity)
  - `Age_Squared` — Non-linear depreciation capture
  - `Brand_Premium` — Direct brand value multiplier as a feature
  - `Power_Weight` — Engine CC per seat (power-to-size ratio)
  - `Log_Km` — Log-transformed km driven (reduces skewness)
- **Feature Scaling** — All 15 features standardized using `sklearn.preprocessing.StandardScaler`
- **Train-Test Split** — 80% training / 20% testing using `sklearn.model_selection.train_test_split`

### 3. Model Training
The following models were trained and compared:

| Model               | R² Score | MAE (Lakhs) |
| ------------------- | -------- | ----------- |
| Random Forest (300 trees) | 0.9263 | ~0.15  |
| Gradient Boosting (500 trees) | 0.9605 | ~0.12 |
| **Ensemble (GBR×0.90 + RF×0.10)** | **0.9598** | **0.1143** |

The **Ensemble model** (weighted average of Gradient Boosting and Random Forest) was selected as the final model.

### 4. Model Evaluation Metrics
- **R² Score** — Measures how well the model explains variance in the target variable (1.0 = perfect). Achieved **96%**
- **MAE (Mean Absolute Error)** — Average absolute difference between predicted and actual prices: **±₹0.11 Lakhs**
- **RMSE (Root Mean Squared Error)** — Penalizes larger errors more heavily
- **5-Fold Cross Validation** — GBR cross-val R² = 0.9549, confirming the model generalizes well
- **Optimal Weight Search** — Tested GBR weights from 0.1 to 0.9 to find the best ensemble blend

### 5. Feature Importance
The trained model identifies which features matter most for price prediction:
- **Car Age** and **Km Driven** are the strongest predictors (depreciation)
- **Brand** has high importance (luxury vs economy)
- **Engine CC** and **Transmission** also contribute significantly

---

## 📊 Features Used for Prediction

| Feature         | Type        | Description                          | Range / Options                     |
| --------------- | ----------- | ------------------------------------ | ----------------------------------- |
| Brand           | Categorical | Car manufacturer                     | Maruti, Hyundai, Honda, Toyota, Ford, Tata, Mahindra, Kia, Volkswagen, BMW, Mercedes, Audi |
| Fuel Type       | Categorical | Type of fuel                         | Petrol, Diesel, CNG, Electric       |
| Transmission    | Categorical | Gearbox type                         | Manual, Automatic                   |
| Seller Type     | Categorical | Who is selling                       | Dealer, Individual, Trustmark Dealer |
| Seats           | Numerical   | Number of seats                      | 4, 5, 6, 7, 8                      |
| Mfg. Year       | Numerical   | Manufacturing year                   | 2000 – 2025                         |
| Km Driven       | Numerical   | Total kilometers driven              | 0 – 5,00,000                        |
| Mileage         | Numerical   | Fuel efficiency in km/l              | 4.0 – 50.0                          |
| Engine CC       | Numerical   | Engine displacement                  | 500 – 5000                          |
| Previous Owners | Numerical   | Number of previous owners            | 1 – 5                               |

### Engineered Features (auto-computed)

| Feature        | Description                                      |
| -------------- | ------------------------------------------------ |
| Car_Age        | 2025 − Manufacturing Year                        |
| Km_per_Year    | Km Driven ÷ Car Age (usage intensity)            |
| Age_Squared    | Car Age² (captures non-linear depreciation)      |
| Brand_Premium  | Brand value multiplier (e.g., Mercedes = 4.8×)   |
| Power_Weight   | Engine CC ÷ Seats (power-to-size ratio)          |
| Log_Km         | log(1 + Km Driven) (reduces skewness)            |

---

## 🚀 How to Run

### Prerequisites

- Python 3.8 or higher
- Required packages:

```bash
pip install numpy pandas scikit-learn
```

> **Note:** `tkinter` comes pre-installed with Python on Windows and macOS. On Linux, install it with:
> ```bash
> sudo apt-get install python3-tk
> ```

### Run the Application

```bash
python car_price_prediction.py
```

**What happens:**
1. The model trains on 10,000 synthetic records (takes 3–5 seconds)
2. A desktop GUI window opens
3. Enter car details using dropdowns and text fields
4. Click **"⚡ PREDICT PRICE"**
5. The predicted price, price range, and market segment are displayed

---

## 🖥️ UI Preview

The application has a dark-themed Tkinter interface with:
- **Dropdown menus** for Brand, Fuel Type, Transmission, Seller Type, and Seats
- **Text input fields** for Year, Km Driven, Mileage, Engine CC, and Owners
- **Predict button** that triggers the ML model
- **Result panel** showing estimated price (₹ Lakhs), price range (±10%), and market segment

### Market Segments
| Price Range     | Segment      |
| --------------- | ------------ |
| Below ₹5 Lakhs | Entry-Level  |
| ₹5 – ₹15 Lakhs | Mid-Range   |
| ₹15 – ₹35 Lakhs | Premium    |
| Above ₹35 Lakhs | Luxury      |

---

## 📈 EDA Visualizations

The `plots/` folder contains 10 charts generated during exploratory data analysis:

| Plot | Description |
| ---- | ----------- |
| `01_target_distribution.png` | Histogram and boxplot of selling prices |
| `02_correlation_heatmap.png` | Correlation matrix of all numerical features |
| `03_price_by_brand.png` | Boxplot of price distribution per brand |
| `04_price_vs_km.png` | Scatter plot — Price vs Km Driven (colored by fuel type) |
| `05_price_vs_year.png` | Scatter plot — Price vs Year (colored by transmission) |
| `06_categorical_counts.png` | Count plots for Brand, Fuel, Transmission, Seller |
| `07_model_r2_comparison.png` | Bar chart comparing R² scores of all 4 models |
| `08_mae_rmse_comparison.png` | Grouped bar chart — MAE and RMSE per model |
| `09_actual_vs_predicted.png` | Scatter plots — Actual vs Predicted for each model |
| `10_feature_importance.png` | Horizontal bar chart of feature importances |

---

## 🛠️ Tech Stack

| Component       | Technology                          |
| --------------- | ----------------------------------- |
| Language        | Python 3.x                          |
| ML Library      | Scikit-Learn                        |
| Models          | Random Forest + Gradient Boosting (Ensemble) |
| Data Handling   | Pandas, NumPy                       |
| Visualization   | Matplotlib, Seaborn                 |
| GUI             | Tkinter                             |
| Model Saving    | Joblib                              |

---

## 📝 ML Pipeline Summary

```
┌──────────────────────────┐
│  1. Generate Data         │  10,000 synthetic car records with realistic pricing
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  2. Feature Engineering   │  5 derived features: Km_per_Year, Age², Brand_Premium, etc.
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  3. Preprocessing         │  Label Encoding → Standard Scaling → 80/20 Split
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  4. Model Training        │  Random Forest (300 trees) + Gradient Boosting (500 trees)
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  5. Ensemble Blending     │  Optimal weight search → GBR×0.90 + RF×0.10
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  6. Evaluation            │  R² = 96%, MAE = ±₹0.11L, 5-Fold CV = 95.5%
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  7. Prediction UI         │  Tkinter GUI — User enters data → Ensemble predicts price
└──────────────────────────┘
```

---

## 🔮 Future Improvements

- Use a real-world dataset (e.g., from Kaggle or CarDekho)
- Add more features like car model name, insurance status, service history
- Deploy as a web application using Flask or Streamlit
- Add hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Implement model persistence so the model doesn't retrain on every launch
- Add data visualization charts directly inside the UI

---

## 👥 Contributors

- Car Price Prediction Team

---

## 📄 License

This project is for educational purposes. Feel free to use and modify.
