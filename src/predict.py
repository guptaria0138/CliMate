"""
predict.py
----------
Simple script: load the trained model + scalers, feed in the last 12 hours
of weather data, and get back the predicted temperature for the next hour.

Usage:
    python src/predict.py --model baseline
"""

import argparse
import pickle
import numpy as np
import tensorflow as tf

from preprocessing import load_raw, split_scale, TIME_STEPS


def predict_next_hour(model, x_scaler, y_scaler, last_12_hours_raw):
    # Scale the input the same way training data was scaled
    scaled_input = x_scaler.transform(last_12_hours_raw)          # (12, n_features)
    scaled_input = scaled_input.reshape(1, TIME_STEPS, -1)        # add batch dimension -> (1, 12, n_features)

    # Predict (output is scaled 0-1, so invert it back to Kelvin)
    pred_scaled = model.predict(scaled_input, verbose=0)
    pred_temp = y_scaler.inverse_transform(pred_scaled)[0][0]
    return pred_temp


def main(model_name="baseline"):
    # Load model + scalers saved by train.py
    model = tf.keras.models.load_model(f"models/{model_name}_model.keras")
    with open(f"models/{model_name}_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    x_scaler, y_scaler = scalers["x_scaler"], scalers["y_scaler"]

    # Example: grab the last 12 hours from the dataset itself (swap this out
    # for live/new data when you actually use it)
    df = load_raw("data/2024-2026.csv")
    last_12_hours_raw = df.drop(columns=["latitude", "longitude", "Temp 2m"]).values[-TIME_STEPS:]

    pred_temp_k = predict_next_hour(model, x_scaler, y_scaler, last_12_hours_raw)
    pred_temp_c = pred_temp_k - 273.15

    print(f"Predicted next-hour temperature: {pred_temp_k:.2f} K  ({pred_temp_c:.2f} °C)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline", choices=["baseline", "attention"])
    args = parser.parse_args()
    main(args.model)