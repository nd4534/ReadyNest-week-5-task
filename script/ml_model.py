import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_taxi_model():
    print("--- Training Big-Data XGBoost Regressor ---")
    
    # Dynamically tracking true file volume
    df = pd.read_csv(r"E:\readynest_task_5\data\cleaned_customer_data.csv")
    total_records = len(df)
    print(f"Loaded database successfully. Ingested Vector Scope: {total_records:,} Rows")
    
    X = df.drop(columns=['fare_amount'])
    y = df['fare_amount']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # High capacity configuration for big data patterns
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,             # Subsampling speeds up training on large data
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',        # Fast histogram method built specifically for millions of rows
        objective='reg:squarederror'
    )
    
    # Dynamic log output reflecting the actual mathematical dataset scale
    print(f"Fitting model rules onto {len(X_train):,} training record entries...")
    model.fit(X_train, y_train)
    
    print("Evaluating test matrix array benchmarks...")
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n=== UPGRADED MODEL RESULTS ===")
    print(f"Total Dataset Points Processed: {total_records:,}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"R-squared Accuracy Score: {r2*100:.2f}%")
    print("==============================\n")
    
    # Save objects
    joblib.dump(model, r'E:\readynest_task_5\data\xgboost_loyalty_model.pkl')
    joblib.dump(list(X.columns), r'E:\readynest_task_5\data\model_features.pkl')
    print("Saved production artifacts successfully.")

if __name__ == "__main__":
    train_taxi_model()