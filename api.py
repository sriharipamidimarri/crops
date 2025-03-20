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
            "Min_Price": min_price,
            "Max_Price": max_price,
            "Modal_Price": modal_price,
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

        # Validate input values exist in encoder classes
        for col in ["State", "District", "Market", "Commodity"]:
            if input_df[col].iloc[0] not in encoders[col].classes_:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid {col} value: '{input_df[col].iloc[0]}'. Must be one of: {', '.join(encoders[col].classes_)}"
                )

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
                "Min_Price": min_price_pred_with_noise[i],
                "Max_Price": max_price_pred_with_noise[i],
                "Modal_Price": modal_price_pred_with_noise[i]
            })

        return {
            "historical_data": historical_data.to_dict(orient="records"),
            "future_predictions": future_predictions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == '__main__':
    app.run(debug=True)
