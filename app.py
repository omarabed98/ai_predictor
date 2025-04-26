# Streamlit app for predicting outputs based on Xf input

import pandas as pd
import streamlit as st
from tensorflow.python.keras import models

# Load the trained model
model = models.load_model("nn_model.keras")

# Page title
st.title("🔮 AI Prediction App")
st.subheader("Enter Xf value to get predictions:")

# User input
user_input = st.number_input("🧮 Enter Xf value", min_value=0.0, step=100.0)

# On button click
if st.button("Predict 🔥"):
    new_input = pd.DataFrame([[user_input]], columns=['Xf'])
    prediction = model.predict(new_input)

    output_columns = ['PR', 'm_dot_f', 'm_dot_b', 'q_e', 'A_e', 'q_c', 'A_c']
    predicted_df = pd.DataFrame(prediction, columns=output_columns)

    # Display results
    st.subheader("🔮 Prediction Results:")
    st.dataframe(predicted_df.T)
