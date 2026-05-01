# =============================================================================
#   CAR PRICE PREDICTION — ML Model with Simple Tkinter UI
#
#   How it works:
#     1. On startup, generates synthetic training data (5000 samples)
#     2. Trains a Gradient Boosting model (~90% R² accuracy)
#     3. Opens a simple GUI where user enters car details
#     4. Predicts the price using the trained ML model
#
#   Run: python car_price_prediction.py
#   Requirements: pip install scikit-learn numpy pandas
# =============================================================================

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import warnings

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Generate Dataset & Train ML Model
# ─────────────────────────────────────────────────────────────────────────────

# Lists used for both training and UI dropdowns
BRANDS        = ["Maruti", "Hyundai", "Honda", "Toyota", "Ford",
                 "Tata", "Mahindra", "Kia", "Volkswagen", "BMW", "Mercedes", "Audi"]
FUEL_TYPES    = ["Petrol", "Diesel", "CNG", "Electric"]
TRANSMISSIONS = ["Manual", "Automatic"]
SELLER_TYPES  = ["Dealer", "Individual", "Trustmark Dealer"]
SEAT_OPTIONS  = [4, 5, 6, 7, 8]

BRAND_PREMIUM = {
    "Maruti": 1.0,  "Hyundai": 1.2,  "Honda": 1.4,  "Toyota": 1.6,
    "Ford": 1.1,    "Tata": 0.95,    "Mahindra": 1.05, "Kia": 1.3,
    "Volkswagen": 1.35, "BMW": 4.2,  "Mercedes": 4.8, "Audi": 3.8,
}


def train_model():
    """Generate synthetic data, train Gradient Boosting model, return model + preprocessors."""
    np.random.seed(42)
    n = 5000

    brand   = np.random.choice(BRANDS, n)
    fuel    = np.random.choice(FUEL_TYPES, n, p=[0.45, 0.35, 0.10, 0.10])
    trans   = np.random.choice(TRANSMISSIONS, n, p=[0.60, 0.40])
    seller  = np.random.choice(SELLER_TYPES, n, p=[0.50, 0.35, 0.15])
    year    = np.random.randint(2005, 2025, n)
    km      = np.random.randint(5000, 200000, n)
    mileage = np.round(np.random.uniform(8.0, 30.0, n), 1)
    seats   = np.random.choice(SEAT_OPTIONS, n, p=[0.05, 0.65, 0.05, 0.15, 0.10])
    owners  = np.random.choice([1, 2, 3, 4], n, p=[0.50, 0.30, 0.15, 0.05])
    engine  = np.random.choice([800, 1000, 1200, 1500, 1800, 2000, 2500, 3000], n)
    car_age = 2025 - year

    # Realistic price formula (in Lakhs) with noise
    price = (
        np.array([BRAND_PREMIUM[b] for b in brand]) * 1.2
        + car_age * -0.18
        + km * -0.00004
        + mileage * 0.04
        + (trans == "Automatic").astype(float) * 0.9
        + (seats - 4) * 0.2
        - (owners - 1) * 0.35
        + (fuel == "Diesel").astype(float) * 0.5
        + (fuel == "Electric").astype(float) * 1.5
        + (seller == "Dealer").astype(float) * 0.2
        + engine * 0.0003
        + np.random.normal(0, 0.4, n)
    )
    price = np.clip(price, 0.5, 100.0).round(2)

    # Build DataFrame
    df = pd.DataFrame({
        "Brand": brand, "Fuel_Type": fuel, "Transmission": trans,
        "Seller_Type": seller, "Km_Driven": km, "Mileage": mileage,
        "Engine_CC": engine, "Seats": seats, "Owners": owners,
        "Car_Age": car_age,
    })

    # Label encode categorical columns
    le_brand = LabelEncoder().fit(BRANDS)
    le_fuel  = LabelEncoder().fit(FUEL_TYPES)
    le_trans = LabelEncoder().fit(TRANSMISSIONS)
    le_sell  = LabelEncoder().fit(SELLER_TYPES)

    df["Brand"]       = le_brand.transform(df["Brand"])
    df["Fuel_Type"]   = le_fuel.transform(df["Fuel_Type"])
    df["Transmission"] = le_trans.transform(df["Transmission"])
    df["Seller_Type"] = le_sell.transform(df["Seller_Type"])

    feature_cols = list(df.columns)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    y = price

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    encoders = {
        "brand": le_brand, "fuel": le_fuel,
        "trans": le_trans, "seller": le_sell,
    }

    return model, encoders, scaler, feature_cols, r2, mae


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Simple Tkinter UI
# ─────────────────────────────────────────────────────────────────────────────

class CarPriceApp:
    def __init__(self, root, model, encoders, scaler, feature_cols, r2, mae):
        self.root = root
        self.model = model
        self.encoders = encoders
        self.scaler = scaler
        self.feature_cols = feature_cols

        self.root.title("Car Price Prediction — ML Model")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)

        # ── Title ──
        title_frame = tk.Frame(root, bg="#1e1e2e", pady=10)
        title_frame.pack(fill="x", padx=20)

        tk.Label(title_frame, text="🚗 Car Price Predictor",
                 font=("Helvetica", 18, "bold"), bg="#1e1e2e", fg="#e8ff47"
                 ).pack(anchor="w")
        tk.Label(title_frame,
                 text=f"ML Model: Gradient Boosting  |  Accuracy (R²): {r2:.1%}  |  Avg Error: ±₹{mae:.2f}L",
                 font=("Helvetica", 9), bg="#1e1e2e", fg="#888"
                 ).pack(anchor="w")

        # ── Separator ──
        tk.Frame(root, bg="#333", height=1).pack(fill="x", padx=20, pady=5)

        # ── Form ──
        form = tk.Frame(root, bg="#1e1e2e", padx=20, pady=5)
        form.pack()

        self.inputs = {}
        row = 0

        # Dropdowns
        dropdowns = [
            ("Brand",        "brand",  BRANDS),
            ("Fuel Type",    "fuel",   FUEL_TYPES),
            ("Transmission", "trans",  TRANSMISSIONS),
            ("Seller Type",  "seller", SELLER_TYPES),
            ("Seats",        "seats",  [str(s) for s in SEAT_OPTIONS]),
        ]
        for label_text, key, options in dropdowns:
            tk.Label(form, text=label_text, font=("Helvetica", 10),
                     bg="#1e1e2e", fg="#ccc", anchor="w", width=18
                     ).grid(row=row, column=0, pady=5, sticky="w")
            var = tk.StringVar(value=options[0])
            cb = ttk.Combobox(form, textvariable=var, values=options,
                              state="readonly", width=20, font=("Helvetica", 10))
            cb.grid(row=row, column=1, pady=5, padx=(10, 0))
            self.inputs[key] = var
            row += 1

        # Number inputs
        number_fields = [
            ("Mfg. Year",       "year",    "2018"),
            ("Km Driven",       "km",      "45000"),
            ("Mileage (km/l)",  "mileage", "18.0"),
            ("Engine CC",       "engine",  "1200"),
            ("Previous Owners", "owners",  "1"),
        ]
        for label_text, key, default in number_fields:
            tk.Label(form, text=label_text, font=("Helvetica", 10),
                     bg="#1e1e2e", fg="#ccc", anchor="w", width=18
                     ).grid(row=row, column=0, pady=5, sticky="w")
            entry = tk.Entry(form, font=("Helvetica", 10), width=22,
                             bg="#2a2a3e", fg="white", insertbackground="white",
                             relief="flat", highlightbackground="#444",
                             highlightthickness=1)
            entry.insert(0, default)
            entry.grid(row=row, column=1, pady=5, padx=(10, 0))
            self.inputs[key] = entry
            row += 1

        # ── Predict Button ──
        btn_frame = tk.Frame(root, bg="#1e1e2e", pady=10)
        btn_frame.pack(fill="x", padx=20)

        tk.Button(btn_frame, text="⚡  PREDICT PRICE", font=("Helvetica", 12, "bold"),
                  bg="#e8ff47", fg="#111", relief="flat", cursor="hand2",
                  padx=20, pady=10, command=self.predict,
                  activebackground="#d4eb3a"
                  ).pack(fill="x")

        # ── Result Area ──
        tk.Frame(root, bg="#333", height=1).pack(fill="x", padx=20, pady=5)

        self.result_frame = tk.Frame(root, bg="#1e1e2e", padx=20, pady=10)
        self.result_frame.pack(fill="x")

        self.result_label = tk.Label(self.result_frame,
                                     text="Enter car details above and click Predict",
                                     font=("Helvetica", 11), bg="#1e1e2e", fg="#666",
                                     justify="left", anchor="w")
        self.result_label.pack(fill="x")

        # ── Footer ──
        tk.Label(root, text="Powered by Scikit-Learn  •  Gradient Boosting Regressor",
                 font=("Helvetica", 8), bg="#1e1e2e", fg="#444", pady=8
                 ).pack(side="bottom")

        # Center window
        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    def predict(self):
        """Read user inputs, preprocess, predict with ML model, show result."""
        try:
            brand   = self.inputs["brand"].get()
            fuel    = self.inputs["fuel"].get()
            trans   = self.inputs["trans"].get()
            seller  = self.inputs["seller"].get()
            seats   = int(self.inputs["seats"].get())
            year    = int(self.inputs["year"].get())
            km      = int(self.inputs["km"].get())
            mileage = float(self.inputs["mileage"].get())
            engine  = int(self.inputs["engine"].get())
            owners  = int(self.inputs["owners"].get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers in all fields.")
            return

        # Validate ranges
        if not (2000 <= year <= 2025):
            messagebox.showwarning("Warning", "Year should be between 2000 and 2025.")
            return
        if not (0 <= km <= 500000):
            messagebox.showwarning("Warning", "Km Driven should be between 0 and 5,00,000.")
            return
        if not (4.0 <= mileage <= 50.0):
            messagebox.showwarning("Warning", "Mileage should be between 4 and 50 km/l.")
            return
        if not (1 <= owners <= 5):
            messagebox.showwarning("Warning", "Owners should be between 1 and 5.")
            return

        car_age = 2025 - year

        # Encode categorical values (same as training)
        row_data = {
            "Brand":        self.encoders["brand"].transform([brand])[0],
            "Fuel_Type":    self.encoders["fuel"].transform([fuel])[0],
            "Transmission": self.encoders["trans"].transform([trans])[0],
            "Seller_Type":  self.encoders["seller"].transform([seller])[0],
            "Km_Driven":    km,
            "Mileage":      mileage,
            "Engine_CC":    engine,
            "Seats":        seats,
            "Owners":       owners,
            "Car_Age":      car_age,
        }

        # Create DataFrame in same column order as training
        X_new = pd.DataFrame([row_data])[self.feature_cols]

        # Scale using the same scaler
        X_scaled = self.scaler.transform(X_new)

        # Predict
        price = max(0.5, self.model.predict(X_scaled)[0])
        low   = price * 0.90
        high  = price * 1.10

        # Market segment
        if price < 5:     segment = "Entry-Level"
        elif price < 15:  segment = "Mid-Range"
        elif price < 35:  segment = "Premium"
        else:             segment = "Luxury"

        # Display result
        result_text = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  💰  Estimated Price :  ₹ {price:.2f} Lakhs\n"
            f"  📉  Price Range     :  ₹ {low:.2f}L  –  ₹ {high:.2f}L\n"
            f"  🏷️   Segment         :  {segment}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  Car  : {brand} | {fuel} | {trans} | {seats} Seater\n"
            f"  Year : {year} ({car_age} yrs old) | {km:,} km\n"
            f"  Info : {mileage} km/l | {engine} CC | {owners} owner(s)"
        )

        self.result_label.config(
            text=result_text,
            font=("Consolas", 10),
            fg="#e8ff47",
            justify="left",
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("⏳ Training ML model... please wait...")
    model, encoders, scaler, feature_cols, r2, mae = train_model()
    print(f"✅ Model ready!  R² = {r2:.2%}  |  MAE = ±₹{mae:.2f}L")
    print("🚀 Launching UI...")

    root = tk.Tk()
    app = CarPriceApp(root, model, encoders, scaler, feature_cols, r2, mae)
    root.mainloop()
