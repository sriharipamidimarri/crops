from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# Paths for saving models and encoders
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Input data model
class PricePredictionInput(BaseModel):
    State: str
    District: str
    Market: str
    Commodity: str
    Arrival_Date: str


# Load dataset
try:
    data = pd.read_csv("data/Price_Agriculture_commodities_Week.csv")
    data["Arrival_Date"] = pd.to_datetime(data["Arrival_Date"])
except Exception as e:
    raise Exception(f"Error loading dataset: {str(e)}")

# Preprocess the dataset
def preprocess_data(data):
    # Extract day, month, year from Arrival_Date
    data["Year"] = data["Arrival_Date"].dt.year
    data["Month"] = data["Arrival_Date"].dt.month
    data["Day"] = data["Arrival_Date"].dt.day

    # Encode categorical variables
    encoders = {}
    for col in ["State", "District", "Market", "Commodity"]:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder

    return data, encoders


# # Train the models
# def train_models():
#     global scaler, rf_min, rf_max, rf_modal

#     # Preprocess data
#     preprocessed_data, encoders = preprocess_data(data)

#     # Features and targets
#     X = preprocessed_data[["State", "District", "Market", "Commodity", "Year", "Month", "Day"]]
#     y_min = preprocessed_data["Min Price"]
#     y_max = preprocessed_data["Max Price"]
#     y_modal = preprocessed_data["Modal Price"]

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, pd.concat([y_min, y_max, y_modal], axis=1), test_size=0.2, random_state=42)
#     y_train_min, y_train_max, y_train_modal = y_train["Min Price"], y_train["Max Price"], y_train["Modal Price"]
#     y_test_min, y_test_max, y_test_modal = y_test["Min Price"], y_test["Max Price"], y_test["Modal Price"]

#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Train models
#     rf_min = RandomForestRegressor(random_state=42)
#     rf_min.fit(X_train_scaled, y_train_min)

#     rf_max = RandomForestRegressor(random_state=42)
#     rf_max.fit(X_train_scaled, y_train_max)

#     rf_modal = RandomForestRegressor(random_state=42)
#     rf_modal.fit(X_train_scaled, y_train_modal)

#     # Save models and scaler
#     joblib.dump(rf_min, os.path.join(MODEL_DIR, "rf_min.pkl"), compress=3)  # Compression Level 3 (balanced)
#     joblib.dump(rf_max, os.path.join(MODEL_DIR, "rf_max.pkl"), compress=3)
#     joblib.dump(rf_modal, os.path.join(MODEL_DIR, "rf_modal.pkl"), compress=3)
#     pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb"))
#     for key, encoder in encoders.items():
#         pickle.dump(encoder, open(os.path.join(MODEL_DIR, f"{key}_encoder.pkl"), "wb"))

#     return X_test_scaled, y_test_min, y_test_max, y_test_modal


# # Initialize models
# train_models()


# Helper function to load models and encoders
def load_models():
    global rf_min, rf_max, rf_modal, scaler, encoders
    rf_min = joblib.load(os.path.join(MODEL_DIR, "rf_min.pkl"))
    rf_max = joblib.load(os.path.join(MODEL_DIR, "rf_max.pkl"))
    rf_modal = joblib.load(os.path.join(MODEL_DIR, "rf_modal.pkl"))
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

    encoders = {}
    for col in ["State", "District", "Market", "Commodity"]:
        encoders[col] = pickle.load(open(os.path.join(MODEL_DIR, f"{col}_encoder.pkl"), "rb"))


load_models()


# API Endpoints
@app.get("/")
def home():
    return {"message": "Welcome to the Price Prediction API!"}


@app.post("/predict")
def predict(input_data: PricePredictionInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Add date features
        input_df["Arrival_Date"] = pd.to_datetime(input_df["Arrival_Date"])
        input_df["Year"] = input_df["Arrival_Date"].dt.year
        input_df["Month"] = input_df["Arrival_Date"].dt.month
        input_df["Day"] = input_df["Arrival_Date"].dt.day

        # Encode categorical features
        for col in ["State", "District", "Market", "Commodity"]:
            input_df[col] = encoders[col].transform(input_df[col])

        # Prepare features
        X_input = input_df[["State", "District", "Market", "Commodity", "Year", "Month", "Day"]]
        X_input_scaled = scaler.transform(X_input)

        # Predict prices
        min_price = rf_min.predict(X_input_scaled)[0]
        max_price = rf_max.predict(X_input_scaled)[0]
        modal_price = rf_modal.predict(X_input_scaled)[0]

        return {
            "Min Price": min_price,
            "Max Price": max_price,
            "Modal Price": modal_price,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
from datetime import datetime, timedelta

# Analysis Endpoint
@app.post("/analysis")
def analysis(input_data: PricePredictionInput):
    """
    Perform analysis using the input fields and provide historical and future price predictions.
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Convert 'Arrival_Date' to datetime
        input_df["Arrival_Date"] = pd.to_datetime(input_df["Arrival_Date"])

        # Extract Year, Month, Day from 'Arrival_Date'
        input_df["Year"] = input_df["Arrival_Date"].dt.year
        input_df["Month"] = input_df["Arrival_Date"].dt.month
        input_df["Day"] = input_df["Arrival_Date"].dt.day

        # Encode categorical features
        for col in ["State", "District", "Market", "Commodity"]:
            input_df[col] = encoders[col].transform(input_df[col])

        # Historical Analysis
        # Encode dataset categorical values (Commodity & Market)
        data_encoded = data.copy()
        for col in ["State", "District", "Market", "Commodity"]:
            data_encoded[col] = encoders[col].transform(data[col])

        # Now filter using encoded values
        historical_data = data_encoded[
            (data_encoded["Commodity"] == input_df["Commodity"].iloc[0]) &
            (data_encoded["Market"] == input_df["Market"].iloc[0])
        ]

        if historical_data.empty:
            raise HTTPException(status_code=404, detail="No historical data found for the given inputs.")

        historical_data = historical_data[["Arrival_Date", "Min Price", "Max Price", "Modal Price"]].tail(10)

        # Future Predictions (next 5 days)
        future_dates = [input_df["Arrival_Date"].iloc[0] + timedelta(days=i) for i in range(1, 6)]
        future_data = pd.DataFrame({
            "State": [input_df["State"].iloc[0]] * len(future_dates),
            "District": [input_df["District"].iloc[0]] * len(future_dates),
            "Market": [input_df["Market"].iloc[0]] * len(future_dates),
            "Commodity": [input_df["Commodity"].iloc[0]] * len(future_dates),
            "Year": [date.year for date in future_dates],
            "Month": [date.month for date in future_dates],
            "Day": [date.day for date in future_dates]
        })

        # Scale features for prediction
        X_future_scaled = scaler.transform(future_data)

        # Predict prices
        min_price_pred = rf_min.predict(X_future_scaled)
        max_price_pred = rf_max.predict(X_future_scaled)
        modal_price_pred = rf_modal.predict(X_future_scaled)

        # Adding noise to predictions
        min_price_pred_with_noise = min_price_pred + np.random.uniform(-5, 5, size=min_price_pred.shape[0])
        max_price_pred_with_noise = max_price_pred + np.random.uniform(-5, 5, size=max_price_pred.shape[0])
        modal_price_pred_with_noise = modal_price_pred + np.random.uniform(-5, 5, size=modal_price_pred.shape[0])

        # Prepare future predictions with noise
        future_predictions = []
        for i, date in enumerate(future_dates):
            future_predictions.append({
                "Arrival_Date": date.strftime("%d-%m-%Y"),
                "Min Price": min_price_pred_with_noise[i],
                "Max Price": max_price_pred_with_noise[i],
                "Modal Price": modal_price_pred_with_noise[i]
            })

        return {
            "historical_data": historical_data.to_dict(orient="records"),
            "future_predictions": future_predictions
        }


    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
if __name__ == '__main__':
    app.run(debug=True)