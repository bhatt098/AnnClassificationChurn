import streamlit as st
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
import pandas as pd
import seaborn as sns
import pickle
from tensorflow.keras.models import load_model


model=load_model('model.h5')

with open('OneHotGeo.pkl','rb') as f:
    encoded_geo=pickle.load(f)

with open('lblEncoderGender.pkl','rb') as f:
    encoded_gender=pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)


##streamlit app
st.title('customer churn prediction')

st.header("Input Customer Features")
credit_score = st.slider("Credit Score", 300, 850, 600)
geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.slider("Age", 18, 70, 30)
tenure = st.slider("Tenure (years)", 1, 10, 5)
balance = st.number_input("Balance", min_value=0.0, max_value=100000.0, value=50000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=30000.0, max_value=120000.0, value=60000.0)



input_data=pd.DataFrame({
    'CreditScore':[credit_score] ,
    'Gender': encoded_gender.transform([gender])[0],
    'Age': [age],
    'Tenure': [tenure],
    'Balance':[balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    })


encoded_geo_val=encoded_geo.transform([[geography]])

geo_encoded_df=pd.DataFrame(encoded_geo_val,columns=encoded_geo.get_feature_names_out(['Geography']))
input_data_df=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

scaled_input_data=scaler.transform(input_data_df)

prediction=model.predict(scaled_input_data)
prediction_probability=prediction[0][0]

if prediction_probability>0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely yo churn')



# import streamlit as st
# from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
# import pandas as pd
# import pickle
# from tensorflow.keras.models import load_model

# # Load your model and encoders
# model = load_model('model.h5')

# with open('OneHotGeo.pkl', 'rb') as f:
#     encoded_geo = pickle.load(f)

# with open('lblEncoderGender.pkl', 'rb') as f:
#     encoded_gender = pickle.load(f)

# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# # Streamlit app
# st.title('Customer Churn Prediction')

# st.header("Input Customer Features")
# credit_score = st.slider("Credit Score", 300, 850, 600)
# geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
# gender = st.selectbox("Gender", ['Male', 'Female'])
# age = st.slider("Age", 18, 70, 30)
# tenure = st.slider("Tenure (years)", 1, 10, 5)
# balance = st.number_input("Balance", min_value=0.0, max_value=100000.0, value=50000.0)
# num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
# has_cr_card = st.selectbox("Has Credit Card", [0, 1])
# is_active_member = st.selectbox("Is Active Member", [0, 1])
# estimated_salary = st.number_input("Estimated Salary", min_value=30000.0, max_value=120000.0, value=60000.0)

# # Prepare input data for prediction
# input_data = pd.DataFrame({
#     'CreditScore': [credit_score],
#     'Gender':[encoded_gender.transform([gender])[0]],
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'NumOfProducts': [num_of_products],
#     'HasCrCard': [has_cr_card],
#     'IsActiveMember': [is_active_member],
#     'EstimatedSalary': [estimated_salary],
# })

# # Transform categorical variables
# geo_encoded_val = encoded_geo.transform([[geography]])
# # gender_encoded_val = encoded_gender.transform([gender]).reshape(-1, 1)

# # Create DataFrames for the encoded features
# geo_encoded_df = pd.DataFrame(geo_encoded_val, columns=encoded_geo.get_feature_names_out())
# # gender_encoded_df = pd.DataFrame(gender_encoded_val, columns=['Gender'])

# # Concatenate encoded features with the input data
# input_data_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# # Ensure input_data_df has all necessary columns for scaling
# scaled_input_data = scaler.transform(input_data_df)

# # Make prediction
# prediction = model.predict(scaled_input_data)
# prediction_probability = prediction[0][0]

# # Display prediction result
# if prediction_probability > 0.5:
#     st.write('The customer is likely to churn')
# else:
#     st.write('The customer is not likely to churn')
