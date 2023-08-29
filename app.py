import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
st.title("Hotel Reservation Cancellation Predictor")
st.markdown('Will the booking be cancelled or not?')

model = open("capstone.pickle", "rb")
clf = pickle.load(model)
model.close()

# Collect user input
no_of_adults = st.slider("Number of Adults", 0, 4)
no_of_children = st.slider("Number of Children", 0, 10)
no_of_weekend_nights = st.slider("Number of Weekend Nights", 0, 7)
no_of_week_nights = st.slider("Number of Week Nights", 1, 17)
type_of_meal_plan = st.selectbox("Type of Meal Plan", ["Not Selected", "Meal Plan 1", "Meal Plan 2", "Meal Plan 3"])
required_car_parking_space = st.selectbox("Required Car Parking Space", ['No', 'Yes'])
room_type_reserved = st.selectbox("Room Type Reserved", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
lead_time = st.slider("Lead Time", 0, 600)
arrival_year = st.selectbox("Arrival Year", [2017, 2018])
arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
arrival_date = st.slider("Arrival Date", 1, 31)
market_segment_type = st.selectbox("Market Segment Type", ["Online", "Offline", "Corporate", "Complementary", "Aviation"])
repeated_guest = st.selectbox("Repeated Guest", ['No', 'Yes'])
no_of_previous_cancellations = st.slider("Number of Previous Cancellations", 0, 13)
no_of_previous_bookings_not_canceled = st.slider("Number of Previous Bookings Not Canceled", 0, 60)
avg_price_per_room = st.slider("Average Price per Room", 0, 540)
no_of_special_requests = st.slider("Number of Special Requests", 0, 5)

# Create a dictionary to hold user input data
input_data = {
    "no_of_adults": [no_of_adults],
    "no_of_children": [no_of_children],
    "no_of_weekend_nights": [no_of_weekend_nights],
    "no_of_week_nights": [no_of_week_nights],
    "type_of_meal_plan": [type_of_meal_plan],
    "required_car_parking_space": [required_car_parking_space],
    "room_type_reserved": [room_type_reserved],
    "lead_time": [lead_time],
    "arrival_year": [arrival_year],
    "arrival_month": [arrival_month],
    "arrival_date": [arrival_date],
    "market_segment_type": [market_segment_type],
    "repeated_guest": [repeated_guest],
    "no_of_previous_cancellations": [no_of_previous_cancellations],
    "no_of_previous_bookings_not_canceled": [no_of_previous_bookings_not_canceled],
    "avg_price_per_room": [avg_price_per_room],
    "no_of_special_requests": [no_of_special_requests]
}

# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

# Perform one-hot encoding on categorical variables
categorical_columns = ["type_of_meal_plan", "room_type_reserved", "market_segment_type",'repeated_guest','required_car_parking_space']
input_df_encoded = pd.get_dummies(input_df, columns=categorical_columns)

# Make the prediction
prediction = clf.predict(input_df_encoded)[0]

if st.button('Prediction'):
    if prediction == 1:
        st.error('Booking will be cancelled')
    else:
        st.success('Booking will not be cancelled')
