import streamlit as st
import pickle
import numpy as np

# Load the pickled model and scaler (assuming they're in the same directory)
try:
    with open("regression.pkl", "rb") as f:
        regmodel = pickle.load(f)
    with open("scaling.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Please ensure 'regression.pkl' and 'scaling.pkl' are in the same directory.")
    exit()

# Title and description for your Streamlit app
st.title("House Price Prediction App")
st.write("Enter the features of a house to predict its price.")

feature_dict = {
                    "CRIM" : "per capita crime rate by town",
                    "ZN" : "proportion of residential land zoned for lots over 25,000 sq.ft",
                    "INDUS" : "proportion of non-retail business acres per town",
                    "CHAS" : "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
                    "NOX" : "nitric oxides concentration (parts per 10 million)",
                    "RM" : "average number of rooms per dwelling",
                    "AGE" : "proportion of owner-occupied units built prior to 1940",
                    "DIS" : "weighted distances to five Boston employment centres",
                    "RAD" : "index of accessibility to radial highways",
                    "TAX" : "full-value property-tax rate per $10,000",
                    "PTRATIO" : "pupil-teacher ratio by town",
                    "B" : "1000(Bk - 0.63)^2 where Bk is the proportion of black people by town",
                    "LSTAT" : "Percentage lower status of the population"
                }


# Create input fields for each feature
feature_values = []
for i in (feature_dict):
    label = f"{i} ({feature_dict[i]})"
    value = st.text_input(label)
    feature_values.append(value)

# Handle missing feature values
if any(val is None for val in feature_values):
    st.error("Error: Please enter values for all features.")
else:
    # Convert input values to floats and handle potential errors
    try:
        input_data = [float(val) for val in feature_values]
    except ValueError:
        st.error("Error: Please enter valid numerical values for features.")
        exit()

    # Reshape the data into a 1D array
    reshaped_data = np.array(input_data).reshape(1, -1)

    # Standardize the data using the scaler
    scaled_data = scaler.transform(reshaped_data)

    # Make prediction
    prediction = regmodel.predict(scaled_data)[0]

    # Display the prediction
    st.success(f"The predicted house price is: ₹{prediction:.2f}")  # Assuming Indian rupees currency
