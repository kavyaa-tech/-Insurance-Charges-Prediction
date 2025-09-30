import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("insurance.csv")
    return data

data = load_data()

st.title("💡 Insurance Charges Prediction (Multiple Linear Regression)")
st.write("This app predicts medical insurance charges based on user inputs.")

# Encode categorical variables
df = data.copy()
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df["sex"] = le_sex.fit_transform(df["sex"])
df["smoker"] = le_smoker.fit_transform(df["smoker"])
df["region"] = le_region.fit_transform(df["region"])

X = df.drop("charges", axis=1)
y = df["charges"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions for evaluation
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Sidebar inputs
st.sidebar.header("Input Features")

age = st.sidebar.slider("Age", int(df.age.min()), int(df.age.max()), 30)
sex = st.sidebar.selectbox("Sex", le_sex.classes_)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.sidebar.slider("Children", int(df.children.min()), int(df.children.max()), 0)
smoker = st.sidebar.selectbox("Smoker", le_smoker.classes_)
region = st.sidebar.selectbox("Region", le_region.classes_)

# Convert categorical inputs to encoded values
sex_encoded = le_sex.transform([sex])[0]
smoker_encoded = le_smoker.transform([smoker])[0]
region_encoded = le_region.transform([region])[0]

input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])

# Button for prediction
if st.button("🔮 Predict Insurance Charges"):
    prediction = model.predict(input_data)[0]
    st.subheader("Predicted Insurance Charges 💰")
    st.write(f"Estimated charges: **${prediction:,.2f}**")

    # Model Performance
    st.subheader("Model Performance")
    st.write(f"Training R²: {model.score(X_train, y_train):.3f}")
    st.write(f"Test R²: {model.score(X_test, y_test):.3f}")
    
    # Error metrics
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    st.write(f"Training RMSE: {rmse_train:,.2f}")
    st.write(f"Test RMSE: {rmse_test:,.2f}")
    st.write(f"Test MAE: {mae_test:,.2f}")
