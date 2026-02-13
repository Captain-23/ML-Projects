import os
import flask
from flask import Flask, render_template

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

app = flask.Flask(__name__)
BASE_DIR = os.path.dirname(__file__)

# Paths to artifacts (assumed to be in the same folder)
MODEL_PATH = os.path.join(BASE_DIR, 'model.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
GENDER_LE_PATH = os.path.join(BASE_DIR, 'label_encoder_gender.pkl')
GEO_OHE_PATH = os.path.join(BASE_DIR, 'onehot_encoder_geo.pkl')

# Try to load artifacts at startup and keep graceful failures
model = None
scaler = None
gender_le = None
geo_ohe = None

def try_load(path, loader):
    try:
        return loader(path)
    except Exception as e:
        print(f'Warning: failed to load {path}: {e}')
        return None

# Load model and transformers
model = try_load(MODEL_PATH, lambda p: tf.keras.models.load_model(p))
scaler = try_load(SCALER_PATH, lambda p: pickle.load(open(p, 'rb')))
gender_le = try_load(GENDER_LE_PATH, lambda p: pickle.load(open(p, 'rb')))
geo_ohe = try_load(GEO_OHE_PATH, lambda p: pickle.load(open(p, 'rb')))

# Define expected features and preprocessing
NUMERIC_COLS = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
CATEGORICAL_GENDER = 'Gender'
CATEGORICAL_GEO = 'Geography'

def preprocess_input(data: dict):
    """Take a dict from the form or JSON and return a feature array ready for the model.
    This function attempts to use saved encoders/scaler; if missing, it will do simple numeric casting.
    """
    # Build single-row DataFrame so we can reuse sklearn transformers easily
    df = pd.DataFrame([{
        'CreditScore': float(data.get('CreditScore', 0)),
        'Geography': data.get('Geography', 'France'),
        'Gender': data.get('Gender', 'Male'),
        'Age': float(data.get('Age', 0)),
        'Tenure': float(data.get('Tenure', 0)),
        'Balance': float(data.get('Balance', 0)),
        'NumOfProducts': float(data.get('NumOfProducts', 1)),
        'HasCrCard': int(data.get('HasCrCard', 1)),
        'IsActiveMember': int(data.get('IsActiveMember', 1)),
        'EstimatedSalary': float(data.get('EstimatedSalary', 0)),
    }])

    # Encode gender
    if gender_le is not None:
        try:
            df['Gender'] = gender_le.transform(df['Gender'])
        except Exception:
            # fallback: map simple
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).fillna(0)
    else:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).fillna(0)

    # Encode geography with one-hot
    geo_features = np.empty((1, 0))
    if geo_ohe is not None:
        try:
            geo_arr = geo_ohe.transform(df[[CATEGORICAL_GEO]]).toarray()
            geo_features = geo_arr
        except Exception:
            # if transform fails, fall back to dummy encoding for up to 3 known countries
            known = ['France', 'Spain', 'Germany']
            geo_features = np.array([[1 if df.at[0, 'Geography'] == k else 0 for k in known]])
    else:
        known = ['France', 'Spain', 'Germany']
        geo_features = np.array([[1 if df.at[0, 'Geography'] == k else 0 for k in known]])

    # Scale numeric columns
    numeric_vals = df[NUMERIC_COLS].values
    if scaler is not None:
        try:
            scaled = scaler.transform(numeric_vals)
        except Exception:
            scaled = numeric_vals
    else:
        scaled = numeric_vals

    # Final ordering: scaled numeric cols + gender + geo one-hot
    gender_col = df[['Gender']].values
    X = np.hstack([scaled, gender_col, geo_features]).astype(np.float32)
    return X

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form values
    payload = {k: flask.request.form.get(k) for k in flask.request.form.keys()}

    X = preprocess_input(payload)

    if model is None:
        # Return informative message if model not loaded
        return render_template('index.html', result={
            'label': 'Model not loaded',
            'probability': 'N/A',
            'risk': 'Unknown'
        })

    try:
        prob = float(model.predict(X)[0][0])
    except Exception as e:
        print(f'Prediction error: {e}')
        return render_template('index.html', result={
            'label': 'Prediction failed',
            'probability': 'N/A',
            'risk': 'Unknown'
        })

    prob_pct = round(prob * 100, 2)
    label = 'Will Churn' if prob >= 0.5 else 'Will Stay'
    risk = 'High' if prob >= 0.5 else 'Low'

    return render_template('index.html', result={'label': label, 'probability': prob_pct, 'risk': risk})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint that accepts JSON and returns JSON with churn probability."""
    if not flask.request.is_json:
        return flask.jsonify({'error': 'Expected JSON body'}), 400

    payload = flask.request.get_json()
    X = preprocess_input(payload)

    if model is None:
        return flask.jsonify({'error': 'Model not available on server'}), 503

    try:
        prob = float(model.predict(X)[0][0])
    except Exception as e:
        return flask.jsonify({'error': f'Prediction failed: {e}'}), 500

    return flask.jsonify({'probability': prob, 'will_churn': prob >= 0.5})

if __name__ == '__main__':
    # Reduce TensorFlow logging (optional)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Disable Flask reloader and threading to avoid loading TensorFlow multiple times
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=False)