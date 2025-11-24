import pandas as pd
import numpy as np
import requests
import os
import glob
import csv
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# --- CONFIGURATION ---
# Based on your screenshots, main.py sits next to the 'data' folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
WEATHER_RAW_DIR = os.path.join(DATA_DIR, 'hourly_weather_raw')
BTS_RAW_DIR = os.path.join(DATA_DIR, 'bts_delay_data')

# Create subdirectories if they don't exist
os.makedirs(WEATHER_RAW_DIR, exist_ok=True)

# Target Airports (Southwest Focus)
TARGET_ICAO = [
    "KDEN", "KLAS", "KMDW", "KBWI", "KDAL", "KATL", "KPHX", "KHOU",
    "KMCO", "KLAX", "KBNA", "KOAK", "KSTL", "KSAN", "KTPA", "KAUS",
    "KFLL", "KMSY", "KSJC", "KSMF", "KBOS", "KMCI", "KSAT", "KSNA",
    "KRDU"
]

TARGET_IATA = [code[1:] for code in TARGET_ICAO] # Removes 'K' for BTS data (e.g., DEN, LAS)

def step_1_process_stations():
    """Filters the ISD history file for target stations."""
    print("\n--- Step 1: Processing Station Metadata ---")
    input_path = os.path.join(DATA_DIR, 'isd-history.csv')
    output_path = os.path.join(DATA_DIR, 'sw_stations.csv')
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please download isd-history.csv first.")
        return None

    df = pd.read_csv(input_path)
    df = df[df['ICAO'].isin(TARGET_ICAO)]
    df = df[['USAF', 'WBAN', 'ICAO', 'STATION NAME']].copy()
    
    # Format IDs ensuring they are strings and zero-padded
    df['USAF'] = df['USAF'].astype(str)
    df['WBAN'] = df['WBAN'].astype(str).str.zfill(5)
    
    df.to_csv(output_path, index=False)
    print(f"Saved filtered stations to {output_path}")
    return df

def step_2_download_weather(stations_df):
    """Downloads NOAA weather data if not already present."""
    print("\n--- Step 2: Downloading Weather Data ---")
    base_url = 'https://www.ncei.noaa.gov/data/global-hourly/access/'
    years = [2022, 2023, 2024]
    
    for index, row in stations_df.iterrows():
        station_id = f"{row['USAF']}{row['WBAN']}"
        icao_code = row['ICAO']
        
        for year in years:
            output_path = os.path.join(WEATHER_RAW_DIR, f"{icao_code}_{year}.csv")
            
            if os.path.exists(output_path):
                print(f"Skipping {icao_code} ({year}) - Already exists.")
                continue
                
            file_url = f"{base_url}{year}/{station_id}.csv"
            try:
                print(f"Downloading {icao_code} ({year})...")
                response = requests.get(file_url)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                else:
                    print(f"Failed (HTTP {response.status_code})")
            except Exception as e:
                print(f"Error downloading: {e}")

def parse_noaa_data(df):
    """Helper function to clean raw NOAA data."""
    # 1. Temperature (TMP)
    temp_str = df['TMP'].astype(str).str.split(',', expand=True)[0]
    df['temp_c'] = pd.to_numeric(temp_str, errors='coerce') / 10.0
    df.loc[df['temp_c'] > 100, 'temp_c'] = np.nan

    # 2. Dew Point
    if 'DEW' in df.columns:
        dew_str = df['DEW'].astype(str).str.split(',', expand=True)[0]
        df['dew_point_c'] = pd.to_numeric(dew_str, errors='coerce') / 10.0
        df.loc[df['dew_point_c'] > 100, 'dew_point_c'] = np.nan
    else:
        df['dew_point_c'] = np.nan

    # 3. Wind Speed
    wnd_str = df['WND'].astype(str).str.split(',', expand=True)[3]
    df['wind_speed_ms'] = pd.to_numeric(wnd_str, errors='coerce') / 10.0
    df.loc[df['wind_speed_ms'] > 100, 'wind_speed_ms'] = np.nan

    # 4. Visibility
    vis_str = df['VIS'].astype(str).str.split(',', expand=True)[0]
    df['visibility_m'] = pd.to_numeric(vis_str, errors='coerce')
    df.loc[df['visibility_m'] > 100000, 'visibility_m'] = np.nan

    # 5. Ceiling
    if 'CIG' in df.columns:
        cig_str = df['CIG'].astype(str).str.split(',', expand=True)[0]
        df['ceiling_m'] = pd.to_numeric(cig_str, errors='coerce')
        df.loc[df['ceiling_m'] == 99999, 'ceiling_m'] = 30000
    else:
        df['ceiling_m'] = np.nan

    # 6. Precip
    if 'AA1' in df.columns:
        precip_str = df['AA1'].astype(str).str.split(',', expand=True)[1]
        df['precip_depth_mm'] = pd.to_numeric(precip_str, errors='coerce') / 10.0
        df.loc[df['precip_depth_mm'] > 1000, 'precip_depth_mm'] = np.nan
        df['precip_depth_mm'] = df['precip_depth_mm'].fillna(0)
    else:
        df['precip_depth_mm'] = 0.0

    # 7. Weather Codes
    df['is_fog'] = 0
    df['is_rain'] = 0
    df['is_snow'] = 0
    df['is_thunder'] = 0

    if 'AW1' in df.columns:
        aw_code = df['AW1'].astype(str).str.split(',', expand=True)[0]
        aw_code = pd.to_numeric(aw_code, errors='coerce')
        df.loc[(aw_code >= 10) & (aw_code <= 49), 'is_fog'] = 1
        df.loc[(aw_code >= 50) & (aw_code <= 69), 'is_rain'] = 1
        df.loc[(aw_code >= 80) & (aw_code <= 84), 'is_rain'] = 1
        df.loc[(aw_code >= 70) & (aw_code <= 79), 'is_snow'] = 1
        df.loc[(aw_code >= 85) & (aw_code <= 89), 'is_snow'] = 1
        df.loc[aw_code >= 90, 'is_thunder'] = 1

    return df[['DATE', 'temp_c', 'dew_point_c', 'wind_speed_ms', 'visibility_m',
               'ceiling_m', 'precip_depth_mm', 'is_fog', 'is_rain', 'is_snow', 'is_thunder']]

def step_3_clean_weather():
    """Reads raw weather files, cleans them, and creates a master weather file."""
    print("\n--- Step 3: Cleaning and Compiling Weather Data ---")
    files = glob.glob(os.path.join(WEATHER_RAW_DIR, "*.csv"))
    
    if not files:
        print("No raw weather files found.")
        return

    all_data = []
    print(f"Found {len(files)} files to process.")
    
    for filename in files:
        try:
            # Extract ICAO code from filename (e.g., KATL_2022.csv -> KATL)
            icao_code = os.path.basename(filename).split('_')[0]
            
            raw_df = pd.read_csv(filename, low_memory=False)
            clean_df = parse_noaa_data(raw_df)
            clean_df["airport_code"] = icao_code
            all_data.append(clean_df)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if all_data:
        master_weather = pd.concat(all_data, ignore_index=True)
        master_weather['timestamp'] = pd.to_datetime(master_weather['DATE']).dt.floor('h')
        master_weather = master_weather.drop(columns=['DATE'])
        
        # Aggregate duplicates (rare but possible)
        final_df = master_weather.groupby(['airport_code', 'timestamp']).mean().reset_index()
        
        # Fill missing values with forward fill
        final_df = final_df.sort_values(by=['airport_code', 'timestamp'])
        final_df = final_df.ffill()
        
        output_path = os.path.join(DATA_DIR, 'weather_master_cleaned.csv')
        final_df.to_csv(output_path, index=False)
        print(f"Saved master weather file: {output_path}")

def step_4_process_flight_data():
    """Unzips and processes BTS flight delay data."""
    print("\n--- Step 4: Processing BTS Flight Data ---")
    
    # Increase CSV limit for large files
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2147483647)

    file_paths = glob.glob(os.path.join(BTS_RAW_DIR, 'T_ONTIME_REPORTING_*.zip'))
    
    if not file_paths:
        print(f"No BTS zip files found in {BTS_RAW_DIR}")
        return

    all_data = []
    columns_to_keep = ['FL_DATE', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'CRS_DEP_TIME', 'ARR_DELAY', 'CANCELLED', 'DIVERTED']

    for file_path in file_paths:
        print(f"Processing {os.path.basename(file_path)}...")
        try:
            df = pd.read_csv(file_path, usecols=columns_to_keep, low_memory=False)
            df = df[df['OP_UNIQUE_CARRIER'] == 'WN'] # Southwest only
            df = df[df['ORIGIN'].isin(TARGET_IATA)] # Target airports only
            all_data.append(df)
        except Exception as e:
            print(f"Error: {e}")

    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(DATA_DIR, 'bts_master_filtered.csv')
        master_df.to_csv(output_path, index=False)
        print(f"Saved filtered flight data: {output_path}")
        return master_df

def step_5_create_disruption_index():
    """Aggregates flight data and calculates Disruption Index via PCA."""
    print("\n--- Step 5: Creating Disruption Index ---")
    input_path = os.path.join(DATA_DIR, 'bts_master_filtered.csv')
    
    if not os.path.exists(input_path):
        print("Flight data not found. Run Step 4 first.")
        return

    df = pd.read_csv(input_path)
    
    # Create Timestamp
    date_col = pd.to_datetime(df['FL_DATE'], errors='coerce')
    date_str = date_col.dt.strftime('%Y-%m-%d')
    time_str = df['CRS_DEP_TIME'].astype(str).str.zfill(4).replace('2400', '2359')
    
    df['timestamp'] = pd.to_datetime(date_str + ' ' + time_str, format='%Y-%m-%d %H%M', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp_h'] = df['timestamp'].dt.floor('h')

    # Aggregate
    grouped = df.groupby(['ORIGIN', 'timestamp_h']).agg(
        avg_arr_delay=('ARR_DELAY', 'mean'),
        percent_cancelled=('CANCELLED', 'mean'),
        percent_diverted=('DIVERTED', 'mean')
    ).reset_index()

    # PCA Calculation
    metrics = grouped[['avg_arr_delay', 'percent_cancelled', 'percent_diverted']]
    scaler = StandardScaler()
    metrics_scaled = scaler.fit_transform(metrics)
    
    pca = PCA(n_components=1)
    grouped['Disruption_Index'] = pca.fit_transform(metrics_scaled)
    
    # Ensure positive correlation (higher index = more disruption)
    if pca.components_[0][0] < 0:
        grouped['Disruption_Index'] = -grouped['Disruption_Index']

    # Normalize column names for merge
    grouped = grouped.rename(columns={'ORIGIN': 'airport_code', 'timestamp_h': 'timestamp'})
    
    output_path = os.path.join(DATA_DIR, 'disruption_index_PCA.csv')
    grouped.to_csv(output_path, index=False)
    print(f"Disruption index saved to {output_path}")

def step_6_train_model():
    """Merges data, engineers features, and trains the model."""
    print("\n--- Step 6: Training XGBoost Model ---")
    weather_path = os.path.join(DATA_DIR, 'weather_master_cleaned.csv')
    disruption_path = os.path.join(DATA_DIR, 'disruption_index_PCA.csv')
    
    if not os.path.exists(weather_path) or not os.path.exists(disruption_path):
        print("Missing input files.")
        return

    df_weather = pd.read_csv(weather_path)
    df_disruption = pd.read_csv(disruption_path)

    # Standardize formats
    df_weather['airport_code'] = df_weather['airport_code'].str.lstrip('K') # KATL -> ATL
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    df_disruption['timestamp'] = pd.to_datetime(df_disruption['timestamp'])

    # Feature Engineering (Lags and Rolling)
    print("Engineering features...")
    df_weather['hour'] = df_weather['timestamp'].dt.hour
    df_weather['month'] = df_weather['timestamp'].dt.month
    df_weather['day_of_week'] = df_weather['timestamp'].dt.dayofweek
    df_weather = df_weather.sort_values(['airport_code', 'timestamp'])

    lag_cols = ['wind_speed_ms', 'visibility_m', 'precip_depth_mm', 'is_thunder', 'is_snow']
    for col in lag_cols:
        df_weather[f'{col}_lag1'] = df_weather.groupby('airport_code')[col].shift(1)
        df_weather[f'{col}_lag3'] = df_weather.groupby('airport_code')[col].shift(3)

    roll_cols = ['wind_speed_ms', 'precip_depth_mm', 'is_snow', 'is_thunder']
    for col in roll_cols:
        df_weather[f'{col}_roll6'] = df_weather.groupby('airport_code')[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
        df_weather[f'{col}_roll12'] = df_weather.groupby('airport_code')[col].transform(lambda x: x.rolling(12, min_periods=1).mean())

    # Merge
    print("Merging datasets...")
    df_master = pd.merge(df_weather, df_disruption, on=['airport_code', 'timestamp'], how='inner')
    
    # Save metadata before One-Hot Encoding
    df_metadata = df_master[['airport_code', 'timestamp']].copy()
    
    # Encoding
    df_master = pd.get_dummies(df_master, columns=['airport_code'], prefix='airport')
    df_master = df_master.dropna()
    df_metadata = df_metadata.loc[df_master.index]

    # Prepare Train/Test
    features = [col for col in df_master.columns if col not in ['Disruption_Index', 'timestamp', 'avg_arr_delay', 'percent_cancelled', 'percent_diverted']]
    target = 'Disruption_Index'
    
    X = df_master[features]
    y = df_master[target]
    
    # Time-series split (no shuffle)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Training on {len(X_train)} records...")
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Eval
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Model RMSE: {rmse:.3f}")
    print(f"Model R2: {r2:.3f}")

    # Save Model
    model_path = os.path.join(DATA_DIR, 'xgb_airline_model.json')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Save Predictions
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(y_train.values.reshape(-1, 1))
    y_pred_scaled = scaler.transform(y_pred.reshape(-1, 1))
    
    results = pd.DataFrame({
        'timestamp': df_metadata.loc[X_test.index]['timestamp'],
        'airport_code': df_metadata.loc[X_test.index]['airport_code'],
        'Raw_Prediction': y_pred,
        'Final_Score_0_100': y_pred_scaled.flatten()
    })
    
    preds_path = os.path.join(DATA_DIR, 'final_model_predictions_local.csv')
    results.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path}")

if __name__ == "__main__":
    # You can comment out steps you've already completed to save time
    
    # Step 1: Filter Stations (Fast)
    #stations = step_1_process_stations()
    
    # Step 2: Download Weather (Slow first time, fast after)
    #if stations is not None:
        #step_2_download_weather(stations)
    
    # Step 3: Clean Weather (Medium)
    #step_3_clean_weather()
    
    # Step 4: Process BTS Zips (Slow - CPU intensive)
    #step_4_process_flight_data()
    
    # Step 5: Disruption Index (Fast)
    #step_5_create_disruption_index()
    
    # Step 6: Train Model (Medium - GPU/CPU intensive)
    step_6_train_model()