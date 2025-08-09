import streamlit as st
import pandas as pd
import joblib
import os

# Path model
MODEL_PATH = "best_model.pkl"  # gunakan .pkl, bukan .sav

# Cek keberadaan model
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' tidak ditemukan. Pastikan sudah di-upload ke repo.")
    st.stop()

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Load data
DATA_PATH = "MEN_SHOES.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset '{DATA_PATH}' tidak ditemukan.")
    st.stop()

data = pd.read_csv(DATA_PATH)

# Application title
st.title('👟 Men Shoes Rating Prediction App')

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    brand_name = st.sidebar.selectbox('Brand Name', sorted(data['Brand_Name'].unique()))
    product_details = st.sidebar.selectbox('Product Details', sorted(data['Product_details'].unique()))
    how_many_sold = st.sidebar.number_input('How Many Sold', min_value=0, value=0)
    current_price = st.sidebar.number_input('Current Price', min_value=0.0, value=0.0, step=0.1)
    
    data_dict = {
        'Brand_Name': brand_name,
        'Product_details': product_details,
        'How_Many_Sold': how_many_sold,
        'Current_Price': current_price
    }
    return pd.DataFrame(data_dict, index=[0])

df = user_input_features()

# Display user inputs
st.subheader('User Input Parameters')
st.write(df)

# Make prediction
if st.button('Predict'):
    try:
        prediction = model.predict(df)
        prediction_value = float(prediction[0])
        st.subheader('Prediction')
        st.write(f'Estimated Rating: {prediction_value:.2f}')
    except Exception as e:
        st.error(f'Error making prediction: {e}')
