import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os
import warnings
warnings.filterwarnings('ignore')

# App title
st.title("Iris Dataset Explorer")
st.write("This app allows you to explore the Iris dataset and make predictions.")

# Load data
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].apply(lambda x: iris.target_names[x])

MODEL_PATH = "iris_model.joblib"

# Check if saved model exists, otherwise train and save it
if os.path.exists(MODEL_PATH):
    model = load(MODEL_PATH)
    st.sidebar.success("Loaded pre-trained model from disk")
else:
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X = df[iris.feature_names]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
    model.fit(X_train, y_train)
    dump(model, MODEL_PATH)
    st.sidebar.success("Trained new model and saved to disk")

# Display data
st.write("### Iris Dataset")
st.dataframe(df)
st.write("### Dataset Description")
st.text(iris.DESCR)
st.write("### Features")
st.write(iris.feature_names)

# Sidebar for user input
st.sidebar.title("Select Features")
sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    float(df['sepal length (cm)'].min()),
    float(df['sepal length (cm)'].max()),
    float(df['sepal length (cm)'].mean())
)
sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    float(df['sepal width (cm)'].min()),
    float(df['sepal width (cm)'].max()),
    float(df['sepal width (cm)'].mean())
)
petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    float(df['petal length (cm)'].min()),
    float(df['petal length (cm)'].max()),
    float(df['petal length (cm)'].mean())
)
petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    float(df['petal width (cm)'].min()),
    float(df['petal width (cm)'].max()),
    float(df['petal width (cm)'].mean())
)

# Make prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display results
st.write("### Prediction Results")
st.subheader(f"Predicted species: **{iris.target_names[prediction][0]}**")

st.write("Prediction probabilities:")
proba_df = pd.DataFrame({
    'Species': iris.target_names,
    'Probability': prediction_proba[0]
})

st.bar_chart(proba_df.set_index('Species'), use_container_width=True)

# Model information
st.sidebar.markdown("---")
st.sidebar.write("Model Information:")
st.sidebar.write(f"Model type: {model.__class__.__name__}")
st.sidebar.write(f"Number of trees: {model.n_estimators}")
st.sidebar.write(f"Classes: {list(iris.target_names)}")