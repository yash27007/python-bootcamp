import streamlit as st
import pandas as pd
st.title("Stremlit text input")

name = st.text_input("Enter your name")

if name:
    st.write(f"Hello {name}")

age = st.slider("Select your age: ",0,100,25)
options=['C','C++','Python','Javascript']
choice = st.selectbox("Choose your favourite language: ",options)
st.write(f"You have selected: {choice}")
st.write(f"Your age is {age}")

data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Helen', 'Ian', 'Julia'],
    'age': [23, 34, 29, 45, 31, 27, 38, 22, 40, 36],
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
}

df = pd.DataFrame(data)
df.to_csv("sample-data.csv")


uploaded_file = st.file_uploader("Choose a CSV file: ",type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)