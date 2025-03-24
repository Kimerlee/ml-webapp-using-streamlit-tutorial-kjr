import streamlit as st
import pandas as pd
import joblib
import numpy as np
import joblib  
import os  

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "global_food_wastage_dataset.pkl")

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

#model = joblib.load("global_food_wastage_dataset.pkl")

# Title of the app
st.title("Global Food Wastage Prediction App")

# Sidebar inputs
st.sidebar.header("Enter Details")

# User inputs
country = st.sidebar.text_input("Enter Country (Type & Press Enter)", "USA") 
year = st.sidebar.number_input("Year", min_value=2018, max_value=2030, value=2024)
population = st.sidebar.number_input("Population (Million)", min_value=1, value=100)
avg_waste_per_capita = st.sidebar.number_input("Avg Waste per Capita (Kg)", min_value=1, value=50)

# Convert country to numerical value (similar to training)
countries = ["Australia", "Indonesia", "Germany", "France", "India", "China", "UK", "South Africa", "Japan", "USA", "Brazil"
             "Saudia Arabia", "Italy", "Spain", "Mexico", "Argentina", "Canada", "South Korea", "Russia", "Turkey"]
if country in countries:
    country_code = countries.index(country)  # Convert to index
else:
    st.sidebar.warning("Country not found in dataset. Using default (0).")
    country_code = 0

# Make prediction
if st.sidebar.button("Predict"):
    input_data = np.array([[country_code, year, population, avg_waste_per_capita]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"Estimated Food Waste: **{prediction:,.2f} Tons**")

# Footer
st.markdown("---")
st.markdown("Built using Streamlit")