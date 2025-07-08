import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Use caching to load the model and columns only once
@st.cache_resource
def load_model_and_columns():
    """Loads the pre-trained model and the required column list."""
    try:
        model = joblib.load('pollution_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except FileNotFoundError:
        st.error("Error: Model or column files not found. Make sure 'pollution_model.pkl' and 'model_columns.pkl' are in the same directory.")
        return None, None

# --- Main App ---
st.set_page_config(page_title="Water Quality Prediction", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS for better UI ---
st.markdown("""
<style>
    /* General Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    /* Title */
    h1 {
        color: #007bff;
        text-align: center;
        font-weight: bold;
    }
    /* Subheader */
    h3 {
        color: #333;
        border-bottom: 2px solid #007bff;
        padding-bottom: 10px;
    }
    /* Form Styling */
    .stForm {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("💧 Water Quality Prediction")
st.write("<p style='text-align: center;'>This app predicts water pollutant levels based on the station ID and year.</p>", unsafe_allow_html=True)
# Load model and data
model, model_columns = load_model_and_columns()
pollutants = ['NH4', 'O2', 'NO3', 'NO2', 'SO4',
       'PO4', 'CL']

if model and model_columns:
    # --- User Input Form ---
    with st.form("prediction_form", border=False):
        st.subheader("Enter Prediction Parameters")
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            station_id_input = st.text_input("Station ID", placeholder="e.g., 22")
        
        with col2:
            # Get the current year as a default
            current_year = datetime.now().year
            year_input = st.number_input("Year", min_value=2000, max_value=current_year + 10, value=current_year)
            
        # Submit button
        submitted = st.form_submit_button("Predict")

    if submitted:
        if not station_id_input:
            st.warning("Please enter a Station ID.")
        else:
            try:
                # --- Prepare Input Data for Model ---
                input_data = pd.DataFrame({'year': [year_input], 'id': [station_id_input]})
                input_encoded = pd.get_dummies(input_data, columns=['id'])
                
                # Align columns with the training data
                missing_cols = set(model_columns) - set(input_encoded.columns)
                for col in missing_cols:
                    input_encoded[col] = 0
                
                # Ensure the order of columns is the same as in the training data
                # Also, handle the case where the user enters a station ID not in the training data
                try:
                    input_encoded = input_encoded[model_columns]
                except KeyError:
                    st.error(f"The station ID '{station_id_input}' was not found in the model's training data. Please use a valid station ID (e.g., 1 to 22).")
                    st.stop() # Stop execution

                # --- Make Prediction ---
                predicted_values = model.predict(input_encoded)[0]
                prediction_results = dict(zip(pollutants, predicted_values))

                # --- Display Results ---
                st.write("---")
                st.subheader(f"Predicted Levels for Station `{station_id_input}` in `{year_input}`")
                
                # Display results in columns using st.metric for a nicer look
                cols = st.columns(len(pollutants)) 
                for i, (pollutant, value) in enumerate(prediction_results.items()):
                    col = cols[i]
                    col.metric(label=pollutant, value=f"{value:.2f}")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
else:
    st.info("The application could not start because the model files are missing.")
