# Real Estate Price Prediction

This project is deployed as an interactive Streamlit application:

https://yingchun-gao-real-estate-price-prediction.streamlit.app/

This application predicts **house prices** based on property characteristics such as size, number of bedrooms, construction year, and other related factors.

The prediction model is a **Random Forest Regressor** trained on historical real estate data.

## Features

- Interactive web interface built with Streamlit
- Property input form for prediction
- Real-time house price estimation
- Feature importance visualization
- Complete machine learning pipeline including:
  - Data cleaning
  - Feature engineering
  - Model training
  - Model evaluation

## Dataset

The dataset includes the following features:

- **Price** – Target variable
- **Year Sold**
- **Property Tax**
- **Insurance**
- **Bedrooms**
- **Bathrooms**
- **Square Footage**
- **Year Built**
- **Lot Size**
- **Basement** (0 = No, 1 = Yes)
- **Property Type** (Bungalow or Condo)

Additional engineered features:

- **Property Age** – Calculated from Year Built and Year Sold
- **Recession Indicator** – Identifies properties sold during 2010–2013
- **Popular Layout Indicator** – Properties with 2 bedrooms and 2 bathrooms

## Technologies Used

- **Streamlit** – Web application interface
- **Scikit-learn** – Random Forest regression model
- **Pandas** – Data preprocessing
- **NumPy** – Numerical operations
- **Matplotlib** – Feature importance visualization