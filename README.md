# 🚗 Car Price Prediction using Machine Learning

A machine learning project that predicts the selling price of used cars based on various features like brand, fuel type, transmission, mileage, engine capacity, and more. The project includes a simple **Tkinter GUI** where users can enter car details and get an instant price prediction powered by a trained **Gradient Boosting Regressor** model.

---

## 📌 Project Overview

| Detail            | Description                                      |
| ----------------- | ------------------------------------------------ |
| **Project Title** | Car Price Prediction using ML                    |
| **Domain**        | Machine Learning — Supervised Learning (Regression) |
| **Algorithm**     | Gradient Boosting Regressor                      |
| **Accuracy (R²)** | ~90%                                            |
| **Avg Error (MAE)** | ±₹0.19 Lakhs                                 |
| **Language**      | Python 3.x                                       |
| **UI**            | Tkinter (Desktop GUI)                            |

---

## 🎯 Objective

Build a machine learning model that can accurately predict the resale/selling price of a used car based on user-provided features. The user enters car details through a simple desktop UI, and the trained ML model returns the estimated price in Indian Rupees (Lakhs).

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
- 5,000 synthetic car records generated using NumPy with realistic distributions
- Features include brand premiums, depreciation curves, fuel-type bonuses, and Gaussian noise to simulate real-world variance

### 2. Data Preprocessing
- **Label Encoding** — Categorical features (Brand, Fuel Type, Transmission, Seller Type) converted to numerical values using `sklearn.preprocessing.LabelEncoder`
- **Feature Scaling** — All features standardized using `sklearn.preprocessing.StandardScaler` (zero mean, unit variance)
- **Train-Test Split** — 80% training / 20% testing using `sklearn.model_selection.train_test_split`

### 3. Model Training
The following models were trained and compared:

| Model               | R² Score | MAE (Lakhs) | RMSE (Lakhs) |
| ------------------- | -------- | ----------- | ------------ |
| Linear Regression   | 0.2969   | 0.6491      | 0.9263       |
| Decision Tree       | 0.7495   | 0.2289      | 0.5529       |
| Random Forest       | 0.8672   | 0.1926      | 0.4026       |
| **Gradient Boosting** | **0.9009** | **0.1886** | **0.3477** |

**Gradient Boosting Regressor** was selected as the best model.

### 4. Model Evaluation Metrics
- **R² Score** — Measures how well the model explains variance in the target variable (1.0 = perfect)
- **MAE (Mean Absolute Error)** — Average absolute difference between predicted and actual prices
- **RMSE (Root Mean Squared Error)** — Penalizes larger errors more heavily than MAE
- **5-Fold Cross Validation** — Ensures the model generalizes well and is not overfitting

### 5. Feature Importance
The trained model identifies which features matter most for price prediction:
- **Car Age** and **Km Driven** are the strongest predictors (depreciation)
- **Brand** has high importance (luxury vs economy)
- **Engine CC** and **Transmission** also contribute significantly

---

## 📊 Features Used for Prediction

| Feature        | Type        | Description                          | Range / Options                     |
| -------------- | ----------- | ------------------------------------ | ----------------------------------- |
| Brand          | Categorical | Car manufacturer                     | Maruti, Hyundai, Honda, Toyota, Ford, Tata, Mahindra, Kia, Volkswagen, BMW, Mercedes, Audi |
| Fuel Type      | Categorical | Type of fuel                         | Petrol, Diesel, CNG, Electric       |
| Transmission   | Categorical | Gearbox type                         | Manual, Automatic                   |
| Seller Type    | Categorical | Who is selling                       | Dealer, Individual, Trustmark Dealer |
| Seats          | Numerical   | Number of seats                      | 4, 5, 6, 7, 8                      |
| Mfg. Year      | Numerical   | Manufacturing year                   | 2000 – 2025                         |
| Km Driven      | Numerical   | Total kilometers driven              | 0 – 5,00,000                        |
| Mileage        | Numerical   | Fuel efficiency in km/l              | 4.0 – 50.0                          |
| Engine CC      | Numerical   | Engine displacement                  | 500 – 5000                          |
| Previous Owners | Numerical  | Number of previous owners            | 1 – 5                               |

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
1. The model trains on 5,000 synthetic records (takes 2–3 seconds)
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
| Data Handling   | Pandas, NumPy                       |
| Visualization   | Matplotlib, Seaborn                 |
| GUI             | Tkinter                             |
| Model Saving    | Joblib                              |

---

## 📝 ML Pipeline Summary

```
┌─────────────────────┐
│  1. Generate Data    │  5000 synthetic car records with realistic pricing
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  2. EDA              │  Statistical analysis + 6 visualization plots
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  3. Preprocessing    │  Label Encoding → Standard Scaling → 80/20 Split
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  4. Model Training   │  Linear Reg, Decision Tree, Random Forest, Gradient Boosting
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  5. Evaluation       │  R², MAE, RMSE, 5-Fold Cross Validation
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  6. Best Model       │  Gradient Boosting (R² = 0.9009)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  7. Prediction UI    │  Tkinter GUI — User enters data → Model predicts price
└─────────────────────┘
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
