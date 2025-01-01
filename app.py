import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the trained model and label encoders (make sure you've saved them)
model = joblib.load('best_model.pkl')  # Path to the trained model
assembly_encoder = joblib.load('assembly_encoder.pkl')  # Path to the assembly label encoder
winner_party_encoder = joblib.load('winner_party_encoder.pkl')  # Path to the winner party label encoder

# Streamlit app UI
st.title("Election Winner Prediction")

st.write("""
    This app predicts the winner of an election assembly based on 
    the number of valid votes, total voters, and the assembly type.
""")

# User inputs
assembly = st.selectbox("Select Assembly", [
            'ADARSH NAGAR',
            'AMBEDKAR NAGAR',
            'BABARPUR',
            'BADARPUR',
            'BADLI',
            'BALLIMARAN',
            'BAWANA',
            'BIJWASAN',
            'BURARI',
            'CHANDNI CHOWK',
            'CHHATARPUR',
            'DELHI CANTT',
            'DEOLI',
            'DWARKA',
            'GANDHI NAGAR',
            'GHONDA',
            'GOKALPUR',
            'GREATER KAILASH',
            'HARI NAGAR',
            'JANAKPURI',
            'JANGPURA',
            'KALKAJI',
            'KARAWAL NAGAR',
            'KAROL BAGH',
            'KASTURBA NAGAR',
            'KIRARI',
            'KONDLI',
            'KRISHNA NAGAR',
            'LAXMI NAGAR',
            'MADIPUR',
            'MALVIYA NAGAR',
            'MANGOL PURI',
            'MATIA MAHAL',
            'MATIALA',
            'MEHRAULI',
            'MODEL TOWN',
            'MOTI NAGAR',
            'MUNDKA',
            'MUSTAFABAD',
            'NAJAFGARH',
            'NANGLOI JAT',
            'NERELA',
            'NEW DELHI',
            'OKHLA',
            'PALAM',
            'PATEL NAGAR',
            'PATPARGANJ',
            'R K PURAM',
            'RAJINDER NAGAR',
            'RAJOURI GARDEN',
            'RITHALA',
            'ROHINI',
            'ROHTAS NAGAR',
            'SADAR BAZAR',
            'SANGAM VIHAR',
            'SEELAMPUR',
            'SEEMA PURI',
            'SHAHDARA',
            'SHAKUR BASTI',
            'SHALIMAR BAGH',
            'SULTANPUR MAJRA',
            'TILAK NAGAR',
            'TIMARPUR',
            'TRI NAGAR',
            'TRILOKPURI',
            'TUGHLAKABAD',
            'UTTAM NAGAR',
            'VIKASPURI',
            'VISHWAS NAGAR',
            'WAZIRPUR'
])
valid_votes = st.number_input("Enter the number of valid votes", min_value=0, step=1)
total_voters = st.number_input("Enter the total number of voters", min_value=0, step=1)

# Predict button
if st.button('Predict Winner'):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'valid_votes': [valid_votes],
        'total_voters': [total_voters],
        'assembly': [assembly]
    })

    # Perform label encoding for the 'assembly' column using assembly_encoder
    input_data['assembly'] = assembly_encoder.transform(input_data['assembly'])

    # Predict the winner
    prediction = model.predict(input_data)

    # Decode the prediction for the winner party using winner_party_encoder
    winner = winner_party_encoder.inverse_transform(prediction)

    st.write(f"The predicted winner is: {winner[0]}")
