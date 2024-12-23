import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('trained_model.sav', 'rb'))

st.title('Iris Prediction')

sepel_len = st.number_input('Sepel Length')
sepel_w = st.number_input('Sepel Width')
petal_len = st.number_input('Petal Length')
petal_w = st.number_input('Petal Width')

input_data = [sepel_len, sepel_w, petal_len, petal_w]


result = ''
if st.button('result'):
    input_data = np.asarray(input_data).reshape(1,-1)
    result = model.predict(input_data)[0]
    
st.success(result)

