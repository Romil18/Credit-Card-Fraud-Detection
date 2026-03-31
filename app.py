
import streamlit as st
import pandas as pd
import joblib

# Load the model and features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
scaler = joblib.load("scaler.pkl") # Load the scaler

st.title("Fraud Detection")

# Input fields for features
inputs_dict = {}
for f in features:
    inputs_dict[f] = st.number_input(f"Enter value for {f}")

if st.button("Predict"):
    # Create a DataFrame from inputs
    df_input = pd.DataFrame([inputs_dict])
    
    # Scale 'Amount' and 'Time' if they are among the selected features
    # Note: The original scaler scaled 'Amount' and 'Time' before feature selection.
    # If these features are still in 'features', we need to re-scale them for new inputs.
    # However, 'Time' and 'Amount' are usually not in `top_features` after feature selection,
    # as V-features are more discriminative. Assuming the model was trained with scaled 'Amount'/'Time',
    # if they are part of 'features', they need to be scaled.
    
    # The top_features list might not contain 'Amount' or 'Time' explicitly,
    # but rather the V-features which are already scaled implicitly if they were part of the initial X.
    # Given `top_features` are 'V14', 'V12', etc., and original Time/Amount columns were dropped before feature selection,
    # the direct scaling of 'Amount' and 'Time' within the app is less straightforward if they are not in `top_features`.
    # For this specific case, `top_features` does *not* contain 'Amount' or 'Time'.
    # If the model was trained with the top 15 V-features, direct scaling of user-inputted 'Amount' and 'Time' isn't needed *if they aren't features themselves*.
    
    # As `top_features` are 'V14', 'V12', 'V17', 'V10', 'V16', 'V4', 'V3', 'V11', 'V2', 'V7', 'V9', 'V21', 'V5', 'V13', 'V1',
    # and these are already processed / scaled through StandardScaler in the training pipeline,
    # the `scaler` loaded here might not be directly applicable to the `inputs_dict` for new predictions
    # unless the Streamlit app asks for raw 'Amount' and 'Time' and then processes them into V-features which is not the current setup.
    # Given the current setup, the app is expecting values for the selected 'V' features directly.
    # Therefore, direct scaling of 'Amount'/'Time' is removed from app.py to match `top_features`.

    pred = model.predict(df_input)

    if pred[0] == 1:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Legitimate Transaction.")
