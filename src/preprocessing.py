import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from datetime import datetime

ENCODERS_PATH = 'models/encoders.pkl'

def basic_cleaning(df):
    # strip column whitespace
    df.columns = [c.strip() for c in df.columns]
    # unify column names
    return df

def feature_engineering(df, reference_date=None):
    # convert dates
    df['Last Purchase Date'] = pd.to_datetime(df['Last Purchase Date'], errors='coerce')

    # set a consistent reference date (use project timeline Aug 01 2025)
    if reference_date is None:
        reference_date = datetime(2025, 8, 1)

    df['DaysSinceLastPurchase'] = (reference_date - df['Last Purchase Date']).dt.days.fillna(9999).astype(int)

    # Create simple derived features
    df['HighValuePurchase'] = (df['Purchase Amount'] > df['Purchase Amount'].median()).astype(int)
    df['FrequentLogin'] = (df['Login Frequency'] >= df['Login Frequency'].median()).astype(int)

    return df

def encode_and_scale(df, fit=True):
    """
    Encodes categorical columns and scales numeric features.
    If fit=True, fit encoders/scaler and save them; otherwise, load and transform.
    """
    categorical = ['Gender', 'Country', 'Product Purchased', 'Membership Status']
    numeric = ['Age', 'Purchase Amount', 'Feedback Score', 'Login Frequency',
               'Customer Support Calls', 'DaysSinceLastPurchase']

    from sklearn.preprocessing import StandardScaler
    encoders = {}
    for col in categorical:
        le = LabelEncoder()
        if fit:
            df[col] = df[col].fillna('NA')
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = joblib.load('models/encoders_{}.pkl'.format(col))
            df[col] = df[col].fillna('NA')
            df[col] = le.transform(df[col])

    # scale numeric features
    scaler = StandardScaler()
    if fit:
        df[numeric] = df[numeric].fillna(df[numeric].median())
        df[numeric] = scaler.fit_transform(df[numeric])
        encoders['scaler'] = scaler
        # save each encoder separately and the scaler
        import joblib, os
        os.makedirs('models', exist_ok=True)
        for k, v in encoders.items():
            joblib.dump(v, f'models/encoders_{k}.pkl')
    else:
        scaler = joblib.load('models/encoders_scaler.pkl')
        df[numeric] = df[numeric].fillna(df[numeric].median())
        df[numeric] = scaler.transform(df[numeric])

    # Save scaler separately for later use
    if fit:
        joblib.dump(scaler, 'models/encoders_scaler.pkl')

    return df

def prepare_for_modeling(path='data/customer_churn.csv', fit=True):
    df = pd.read_csv(path)
    df = basic_cleaning(df)
    df = feature_engineering(df)
    # preserve identifiers for exports
    ids = df[['Customer ID', 'Name']].copy() if 'Customer ID' in df.columns else None
    # drop columns not used for modeling
    drop_cols = ['Customer ID', 'Name', 'Last Purchase Date']
    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df_model = encode_and_scale(df_model, fit=fit)
    return df_model, ids
