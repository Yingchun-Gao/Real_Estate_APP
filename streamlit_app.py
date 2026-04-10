import streamlit as st
import pickle
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(page_title="Real Estate Price Predictor")

st.title("Real Estate Price Predictor")

MODEL_PATH = "models/RFmodel.pkl"

# Check if trained model exists
if not os.path.exists(MODEL_PATH):
    logger.error("Model not found")
    st.error("Model not found. Please run main.py first.")
    st.stop()

try:
    # Load trained model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    logger.error(f"Model loading error: {e}")
    st.error("Failed to load model")
    st.stop()

st.subheader("Enter Property Details")

with st.form("input_form"):

    year_sold = st.number_input("Year Sold", 2000, 2030, 2020)
    year_built = st.number_input("Year Built", 1900, 2030, 2000)

    property_tax = st.number_input("Property Tax", min_value=0)
    insurance = st.number_input("Insurance", min_value=0)

    beds = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
    baths = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6])

    sqft = st.number_input("Square Footage", min_value=200)
    lot_size = st.number_input("Lot Size", min_value=0)

    basement = st.selectbox("Basement", [0, 1])
    property_type = st.selectbox("Property Type", ["Bungalow", "Condo"])

    submit = st.form_submit_button("Predict Price")

# Make prediction
if submit:
    try:
        if year_built > year_sold:
            st.error("Year Built must be earlier than Year Sold.")
            st.stop()

        # Feature engineering (same logic used in data preprocessing)
        property_age = year_sold - year_built
        recession = 1 if 2010 <= year_sold <= 2013 else 0
        popular = 1 if beds == 2 and baths == 2 else 0
        property_type_Condo = 1 if property_type == "Condo" else 0

        input_data = pd.DataFrame(
            [
                {
                    "year_sold": year_sold,
                    "property_tax": property_tax,
                    "insurance": insurance,
                    "beds": beds,
                    "baths": baths,
                    "sqft": sqft,
                    "year_built": year_built,
                    "lot_size": lot_size,
                    "basement": basement,
                    "popular": popular,
                    "recession": recession,
                    "property_age": property_age,
                    "property_type_Condo": property_type_Condo,
                }
            ]
        )

        prediction = model.predict(input_data)[0]

        st.success(f"Estimated Price: ${int(prediction):,}")
        logger.info("Prediction successful")

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error("Prediction failed")
