import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from joblib import load
import os

# App configuration
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
st.title("🌸 Iris Flower Classifier")
st.write("Predict iris flower species using machine learning")

# Load data (for display only)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[x] for x in iris.target]

# Load pre-trained model
@st.cache_resource
def load_model():
    try:
        model = load('iris_model.joblib')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model = load_model()

# Sidebar inputs
st.sidebar.header("Input Features")
inputs = {}
for feature in iris.feature_names:
    inputs[feature] = st.sidebar.slider(
        f"{feature} (cm)",
        float(df[feature].min()),
        float(df[feature].max()),
        float(df[feature].mean()),
        step=0.1
    )

# Prediction
if st.sidebar.button("Predict"):
    input_data = [[inputs[feature] for feature in iris.feature_names]]
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Display results
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Species", iris.target_names[prediction])
    with col2:
        st.metric("Confidence", f"{max(probabilities)*100:.1f}%")
    
    # Probability chart
    prob_df = pd.DataFrame({
        "Species": iris.target_names,
        "Probability": probabilities
    }).set_index("Species")
    st.bar_chart(prob_df)

# Dataset explorer
st.header("Dataset Explorer")
tab1, tab2 = st.tabs(["Raw Data", "Description"])
with tab1:
    st.dataframe(df.sample(5))  # Show random samples
with tab2:
    st.text(iris.DESCR)

# Model info
st.sidebar.header("Model Information")
st.sidebar.write(f"Algorithm: {model.__class__.__name__}")
st.sidebar.write(f"Version: 1.0")