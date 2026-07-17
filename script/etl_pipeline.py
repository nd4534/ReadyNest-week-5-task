import pandas as pd
import numpy as np
import os

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367 * c

def run_streamed_etl(raw_path, output_csv_path, total_rows=3000000, chunk_size=500000):
    print(f"--- Starting Scaled ETL Pipeline ({total_rows:,} Rows) ---")
    
    # Housekeeping: Clear out old run outputs to avoid appending to legacy files
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)
        
    rows_processed = 0
    
    # Streams raw data in chunks directly from your raw training zip archive
    for chunk in pd.read_csv(raw_path, chunksize=chunk_size, parse_dates=['pickup_datetime'], compression='zip'):
        if rows_processed >= total_rows:
            break
            
        print(f"Processing batch {rows_processed // chunk_size + 1}...")
        
        # 1. LOOSENED Operational Boundary Cleaning
        # Widened spatial bounds to keep outer boroughs/airports and kept logical price caps
        chunk = chunk[
            (chunk['fare_amount'] >= 0.0) & (chunk['fare_amount'] < 500) &
            (chunk['passenger_count'] >= 0) & (chunk['passenger_count'] <= 9) &
            (chunk['pickup_longitude'] > -76) & (chunk['pickup_longitude'] < -71) &
            (chunk['pickup_latitude'] > 39) & (chunk['pickup_latitude'] < 43) &
            (chunk['dropoff_longitude'] > -76) & (chunk['dropoff_longitude'] < -71) &
            (chunk['dropoff_latitude'] > 39) & (chunk['dropoff_latitude'] < 43)
        ].dropna()
        
        # 2. Geometry calculations
        chunk['trip_distance_km'] = haversine_np(
            chunk['pickup_longitude'], chunk['pickup_latitude'],
            chunk['dropoff_longitude'], chunk['dropoff_latitude']
        )
        
        # Airport Vectors
        chunk['jfk_dist'] = haversine_np(chunk['dropoff_longitude'], chunk['dropoff_latitude'], -73.7781, 40.6413)
        chunk['lga_dist'] = haversine_np(chunk['dropoff_longitude'], chunk['dropoff_latitude'], -73.8740, 40.7769)
        
        # 3. Temporal cycles
        chunk['hour'] = chunk['pickup_datetime'].dt.hour
        chunk['day_of_week'] = chunk['pickup_datetime'].dt.dayofweek
        chunk['is_rush_hour'] = ((chunk['hour'].isin([7,8,9,16,17,18,19])) & (chunk['day_of_week'] < 5)).astype(int)
        
        # Drop heavy components
        chunk = chunk.drop(columns=['key', 'pickup_datetime'])
        
        # Append rows straight to your single, final master CSV file
        chunk.to_csv(
            output_csv_path, 
            mode='a', 
            index=False, 
            header=not os.path.exists(output_csv_path)
        )
        
        rows_processed += chunk_size
        
    print(f"[SUCCESS] Scaled ETL complete! Master CSV built at {output_csv_path}\n")

if __name__ == "__main__":
    run_streamed_etl(
        raw_path = r"E:\readynest_task_5\data\train.zip",
        output_csv_path = r"E:\readynest_task_5\data\cleaned_customer_data.csv",
        total_rows=3000000
    )