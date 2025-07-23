import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os
import warnings
warnings.filterwarnings('ignore')


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
    

