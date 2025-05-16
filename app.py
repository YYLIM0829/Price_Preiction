from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/env')
def get_env_variables():
    env_vars = {
        "FLASK_ENV": os.getenv("FLASK_ENV", "development"),
        "SECRET_KEY": os.getenv("SECRET_KEY", "not-set")
    }
    return jsonify(env_vars)
    
@app.route('/')
def home():
    return "Price Prediction API is running. Use POST /predict to get predictions."

# Festival info for adjustment
festivals = {
    'Chinese New Year': {'month': 2, 'effect': 'up', 'range': (5, 15)},
    'Labour Day': {'month': 5, 'effect': 'down', 'range': (10, 30)},
    'Hari Raya Aidilfitri': {'month': 4, 'effect': 'down', 'range': (10, 30)},
    'Father\'s Day': {'month': 6, 'effect': 'down', 'range': (10, 30)},
    'Mother\'s Day': {'month': 5, 'effect': 'down', 'range': (10, 30)},
    'Deepavali': {'month': 11, 'effect': 'down', 'range': (10, 25)},
    'Christmas': {'month': 12, 'effect': 'down', 'range': (10, 20)},
    'Merdeka Day': {'month': 8, 'effect': 'down', 'range': (5, 15)},
    '11.11 Sale': {'month': 11, 'effect': 'down', 'range': (20, 50)},
    'Black Friday': {'month': 11, 'effect': 'down', 'range': (30, 60)},
    'Year-End Sale': {'month': 12, 'effect': 'down', 'range': (15, 40)},
}

def check_festival(date):
    month = date.month
    for fest, info in festivals.items():
        if month == info['month']:
            return info
    return None

# Feature creation helper
def create_features(df, window=7):
    X, y = [], []
    for i in range(len(df) - window):
        X.append(df['Final Price'].iloc[i:i+window].values)
        y.append(df['Final Price'].iloc[i+window])
    return np.array(X), np.array(y)

# Evaluation function
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return mae

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    product_name = data.get('product_name')

    if not product_name:
        return jsonify({'error': 'Please provide product_name in JSON'}), 400

    # Load and preprocess data
    df = pd.read_csv('product_daily_prices_2022_2025_monthly_changes.csv')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)


    product_data = df[df['Product'] == product_name].copy()

    if product_data.empty:
        return jsonify({'error': f'Product "{product_name}" not found'}), 404

    product_data = product_data.sort_values('Date')

    X, y = create_features(product_data)

    # Train/test split (no shuffle to maintain time order)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Train models
    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_mae = evaluate(y_test, knn_pred)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae = evaluate(y_test, rf_pred)

    # Prepare data for LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(X_train_lstm.shape[1], 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=16, verbose=0)

    lstm_pred = lstm_model.predict(X_test_lstm).flatten()
    lstm_mae = evaluate(y_test, lstm_pred)

    # Pick best model (lowest MAE)
    maes = [knn_mae, rf_mae, lstm_mae]
    models = ['KNN', 'Random Forest', 'LSTM']
    best_index = np.argmin(maes)
    best_model_name = models[best_index]

    # Prepare to predict next 7 days
    last_window = product_data['Final Price'].iloc[-7:].values.reshape(1, -1)
    next_prices = []

    if best_model_name == 'KNN':
        current_window = last_window.copy()
        for _ in range(7):
            pred = knn.predict(current_window)[0]
            next_prices.append(pred)
            current_window = np.append(current_window[:,1:], [[pred]], axis=1)
    elif best_model_name == 'Random Forest':
        current_window = last_window.copy()
        for _ in range(7):
            pred = rf.predict(current_window)[0]
            next_prices.append(pred)
            current_window = np.append(current_window[:,1:], [[pred]], axis=1)
    else:
        current_window = last_window.reshape((1, 7, 1))
        for _ in range(7):
            pred = lstm_model.predict(current_window).flatten()[0]
            next_prices.append(pred)
            new_window = np.append(current_window.flatten()[1:], [pred])
            current_window = new_window.reshape((1, 7, 1))

    # Apply festival adjustments
    start_date = product_data['Date'].max() + timedelta(days=1)
    predicted_dates = [start_date + timedelta(days=i) for i in range(7)]
    adjusted_prices = []

    for date, price in zip(predicted_dates, next_prices):
        festival = check_festival(date)
        if festival:
            range_min, range_max = festival['range']
            adjustment = np.random.uniform(range_min, range_max)
            if festival['effect'] == 'up':
                price *= (1 + adjustment / 100)
            else:
                price *= (1 - adjustment / 100)
        adjusted_prices.append(price)

    # Format results for JSON
    results = []
    for d, p in zip(predicted_dates, adjusted_prices):
        results.append({
            'date': d.strftime('%Y-%m-%d'),
            'predicted_price': round(float(p), 2)
        })

    return jsonify({
        'product': product_name,
        'best_model': best_model_name,
        'predictions': results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
