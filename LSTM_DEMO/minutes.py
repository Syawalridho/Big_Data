# Nama File: prediksi_10menit.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

FREQ_CODE = '10T'
LOOK_BACK = 36 # Menggunakan 360 menit (6 jam) terakhir untuk prediksi
N_FUTURE = 60  # Memprediksi 60 langkah (600 menit / 10 jam) ke depan
FOLDER_NAME = 'hasil_grafik_10menit'

if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)

def load_and_prepare_data(filepath='data_fixed.csv', freq_code='10T'):
    """Memuat dan mempersiapkan data per 10 menit."""
    print("--- 1. Memuat Data untuk Frekuensi 10 Menit ---")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.replace(' ', '_')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['Populasi'] = pd.to_numeric(df['Populasi'].astype(str).str.replace(',', ''), errors='coerce')
    df.dropna(inplace=True)
    df = df.sort_values('timestamp')

    df_resampled = df.set_index('timestamp').resample(freq_code).agg(Populasi=('Populasi', 'last'))
    df_resampled.interpolate(method='linear', inplace=True)
    return df_resampled

def create_lstm_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def build_and_train_lstm(X_train, y_train, X_val, y_val, look_back):
    """Membangun dan melatih model LSTM."""
    print("--- 2. Membangun dan Melatih Model LSTM ---")
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Melatih model...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=0)
    print("Pelatihan model selesai.\n")
    return model, history

def plot_loss_curve(history):
    """Menampilkan grafik training vs validation loss."""
    print("--- 3. Menampilkan Kurva Loss ---")
    plt.figure(figsize=(12, 7))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Kurva Training vs Validation Loss (10 Menit)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{FOLDER_NAME}/1_kurva_loss.png')
    plt.show()

def plot_full_comparison_and_metrics(model, full_scaled_data, scaler, look_back):
    """Menampilkan perbandingan prediksi vs aktual dari awal hingga akhir dan metrik error."""
    print("--- 4. Menampilkan Perbandingan & Metrik Error ---")
    X_full, y_full = create_lstm_dataset(full_scaled_data, look_back)
    X_full = np.reshape(X_full, (X_full.shape[0], X_full.shape[1], 1))

    full_predict = model.predict(X_full)
    
    full_predict_inv = scaler.inverse_transform(full_predict)
    y_full_inv = scaler.inverse_transform(y_full.reshape(-1, 1))

    mse = mean_squared_error(y_full_inv, full_predict_inv)
    mae = mean_absolute_error(y_full_inv, full_predict_inv)
    print(f"Mean Squared Error (MSE) di seluruh data: {mse:,.0f}")
    print(f"Mean Absolute Error (MAE) di seluruh data: {mae:,.0f} jiwa")

    plt.figure(figsize=(15, 8))
    plt.plot(df.index, scaler.inverse_transform(full_scaled_data), label='Data Aktual Lengkap')
    plt.plot(df.index[look_back:], full_predict_inv, color='red', linestyle='--', label='Prediksi LSTM pada Data Aktual')
    plt.title('Hasil Prediksi vs Hasil Aktual (dari Awal hingga Akhir)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{FOLDER_NAME}/2_prediksi_vs_aktual_lengkap.png')
    plt.show()

def plot_future_forecast_and_increase(model, full_scaled_data, scaler, look_back, original_dates):
    """Membuat prediksi masa depan dan menghitung kenaikannya."""
    print(f"\n--- 5. Membuat Prediksi {N_FUTURE*10} Menit ke Depan ---")
    
    last_sequence = full_scaled_data[-look_back:]
    current_sequence = last_sequence.reshape((1, look_back, 1))
    
    future_predictions = []
    for _ in range(N_FUTURE):
        next_pred_scaled = model.predict(current_sequence, verbose=0)[0]
        future_predictions.append(next_pred_scaled)
        next_pred_reshaped = next_pred_scaled.reshape(1, 1, 1)
        current_sequence = np.append(current_sequence[:, 1:, :], next_pred_reshaped, axis=1)

    future_predictions_inv = scaler.inverse_transform(future_predictions)

    last_population = scaler.inverse_transform(full_scaled_data)[-1][0]
    final_prediction = future_predictions_inv[-1][0]
    total_increase = final_prediction - last_population
    print(f"Total kenaikan populasi dalam {N_FUTURE*10} menit ke depan diprediksi sebesar: +{total_increase:,.0f} jiwa")

    last_date = original_dates[-1]
    future_dates = pd.to_datetime([last_date + pd.Timedelta(minutes=10 * i) for i in range(1, N_FUTURE + 1)])
    
    plt.figure(figsize=(15, 8))
    plt.plot(original_dates, scaler.inverse_transform(full_scaled_data), label='Data Historis')
    plt.plot(future_dates, future_predictions_inv, 'o--', color='red', label=f'Ramalan {N_FUTURE*10} Menit ke Depan')
    plt.title('Ramalan Populasi Masa Depan (per 10 Menit)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{FOLDER_NAME}/3_prediksi_masa_depan.png')
    plt.show()

if __name__ == "__main__":
    df = load_and_prepare_data(freq_code=FREQ_CODE)
    if df is not None:
        population_data = df['Populasi'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(population_data)
        
        train_size = int(len(scaled_data) * 0.9)
        train_data, val_data = scaled_data[0:train_size], scaled_data[train_size - LOOK_BACK:]
        
        X_train, y_train = create_lstm_dataset(train_data, LOOK_BACK)
        X_val, y_val = create_lstm_dataset(val_data, LOOK_BACK)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
        
        model, history = build_and_train_lstm(X_train, y_train, X_val, y_val, LOOK_BACK)
        
        plot_loss_curve(history)
        plot_full_comparison_and_metrics(model, scaled_data, scaler, LOOK_BACK)
        plot_future_forecast_and_increase(model, scaled_data, scaler, LOOK_BACK, df.index)
