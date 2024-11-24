import boto3
import pandas as pd
from io import BytesIO
import os
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
import boto3
from botocore.exceptions import NoCredentialsError
import joblib


# S3 Operations
def get_s3_client():
    return boto3.client('s3')

def load_parquet_from_s3(bucket_name, file_key):
    s3_client = get_s3_client()
    parquet_object = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    return pd.read_parquet(BytesIO(parquet_object['Body'].read()))


def list_parquet_files(bucket_name, prefix=""):
    s3_client = get_s3_client()
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.parquet')]


def get_parquet_files(bucket_name, prefix=""):
    parquet_files = list_parquet_files(bucket_name, prefix)
    print(f"Parquet files found: {parquet_files}")
    
    # Check if any parquet files exist
    if not parquet_files:
        print("No existing parquet files found. Uploading new DataFrame directly.")
        return pd.DataFrame()  # Return an empty DataFrame if no files are found

    all_dataframes = []
    
    # Load existing parquet files into DataFrames
    for file_key in parquet_files:
        df = load_parquet_from_s3(bucket_name, file_key)
        print(f"Loaded {file_key} with columns: {df.columns} and shape: {df.shape}")
        
        # Check for 'id' column presence
        if 'id' not in df.columns:
            print(f"Warning: {file_key} does not contain 'id' column.")
            continue  # Skip this DataFrame if it lacks the 'id' column
        
        all_dataframes.append(df)
    
    # Concatenate all DataFrames into a single DataFrame
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Combined DataFrame shape: {final_df.shape}")
        return final_df
    else:
        print("No valid DataFrames found. Returning empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if no valid DataFrames
    
def train_and_load_model():
    bucket_name = "francejobdata"
    prefix=""
    
    all_dataframes = get_parquet_files(bucket_name, prefix)
    df = all_dataframes.copy()
    df = pd.DataFrame(df)
    # Handle missing target values
    df = df.dropna(subset=['avg_salary'])
    df = df.dropna(subset=['experience'])

    X = df[['job_category', 'experience']]
    y = df['avg_salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = ['job_category']
    numerical_features = ['experience'] #

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    numerical_imputer = SimpleImputer(strategy='mean')

    categorical_transformer = Pipeline(steps=[
        ('imputer', categorical_imputer),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', numerical_imputer),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ]
    )
    model = RandomForestRegressor(random_state=42, n_estimators=100)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    
    model_filename = 'salary_prediction_model.joblib'
    joblib.dump(pipeline, model_filename)
    
    model_filename = 'salary_prediction_model.joblib'
    s3_model_path = 'models/salary_prediction_model.joblib'  # Path within S3

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Upload the model to S3
    try:
        s3.upload_file(model_filename, bucket_name, s3_model_path)
        print(f"Model uploaded successfully to {s3_model_path}")
    except NoCredentialsError:
        print("Credentials not available")