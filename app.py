import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from google.cloud import firestore
import pytz
LOCAL_TZ = pytz.timezone('Asia/Kuala_Lumpur')
import requests
import io
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import holidays
from ipywidgets import interact, Dropdown, DatePicker, Output
from IPython.display import display
from google.oauth2 import service_account
import os
from flask import Flask, request, jsonify
from github import Github

app = Flask(__name__)

if os.path.exists(FIREBASE_KEY_PATH):
    cred = service_account.Credentials.from_service_account_file(FIREBASE_KEY_PATH)
    db = firestore.Client(credentials=cred)
else:
    print("Firebase key file not found. Some functions will not work.")
    db = None


FESTIVAL_DISCOUNTS = {
    'Chinese New Year': {
        'discount': {'gifts': (0.10, 0.20), 'electronics': (0.10, 0.20)},
        'price_increase': {'food': (0.05, 0.15), 'beverages': (0.05, 0.15)},
        'duration': 7
    },
    'Labour Day': {
        'discount': {'all': (0.05, 0.15)},
        'duration': 1
    },
    'Hari Raya': {
        'discount': {'food': (0.10, 0.30), 'clothing': (0.10, 0.30)},
        'duration': 7
    },
    'Father\'s Day': {
        'discount': {'all': (0.05, 0.15)},
        'duration': 2
    },
    'Mother\'s Day': {
        'discount': {'all': (0.05, 0.15)},
        'duration': 2
    },
    'Deepavali': {
        'discount': {'electronics': (0.10, 0.25), 'food': (0.10, 0.25)},
        'duration': 7
    },
    'Christmas': {
        'discount': {'gifts': (0.10, 0.20), 'electronics': (0.10, 0.20)},
        'duration': 7
    },
    'National Day': {
        'discount': {'all': (0.05, 0.15)},
        'duration': 1
    },
    '11.11 Sale': {
        'discount': {'all': (0.20, 0.50)},
        'duration': 1
    },
    'Black Friday': {
        'discount': {'electronics': (0.30, 0.60), 'fashion': (0.30, 0.60)},
        'duration': 1
    },
    'Year-End Sale': {
        'discount': {'clothing': (0.15, 0.40), 'appliances': (0.15, 0.40)},
        'duration': 14
    }
}

malaysia_holidays = holidays.Malaysia()

def is_festival_period(date):
    date_str = date.strftime('%Y-%m-%d')

    festivals = {
        'Chinese New Year': [(1, 22), (1, 23)],
        'Labour Day': [(5, 1)],
        'Hari Raya': [(4, 21)],
        'Father\'s Day': [(6, 18)],
        'Mother\'s Day': [(5, 14)],
        'Deepavali': [(10, 24)],
        'Christmas': [(12, 25)],
        'National Day': [(8, 31)],
        '11.11 Sale': [(11, 11)],
        'Black Friday': [(11, 25)],
        'Year-End Sale': [(12, 15)]
    }

    current_year = date.year
    month_day = (date.month, date.day)

    for festival, dates in festivals.items():
        for m, d in dates:
            festival_date = datetime(current_year, m, d).date()
            duration = FESTIVAL_DISCOUNTS[festival]['duration']
            end_date = festival_date + timedelta(days=duration-1)

            if festival_date <= date.date() <= end_date:
                return festival, festival_date

    return None, None

def get_festival_discount(types, festival):
    if festival not in FESTIVAL_DISCOUNTS:
        return 0

    festival_data = FESTIVAL_DISCOUNTS[festival]

    if 'discount' in festival_data:
        for discount_types, discount_range in festival_data['discount'].items():
            if discount_types == 'all' or discount_types in types.lower():
                return np.random.uniform(*discount_range)

    return 0

def get_festival_price_increase(types, festival):
    if festival not in FESTIVAL_DISCOUNTS:
        return 0

    festival_data = FESTIVAL_DISCOUNTS[festival]

    if 'price_increase' in festival_data:
        for inc_types, inc_range in festival_data['price_increase'].items():
            if inc_types in types.lower():
                return np.random.uniform(*inc_range)

    return 0

def load_data_from_firestore():
    products_ref = db.collection('products')
    docs = products_ref.stream()

    data = []
    for doc in docs:
        product = doc.to_dict()
        last_updated = product.get('last_updated')

        if last_updated and last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=LOCAL_TZ)

        data.append({
            'name': product.get('name', ''),
            'type': product.get('type', ''),
            'price': product.get('price', 0),
            'future_price': product.get('future_price', 0),
            'last_updated': last_updated or datetime.now(LOCAL_TZ)
        })

    return pd.DataFrame(data)

def load_historical_data():
    try:
        response = requests.get(GITHUB_RAW_URL)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))

        date_formats = ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d']
        for fmt in date_formats:
            try:
                df['date'] = pd.to_datetime(df['date'], format=fmt, errors='raise')
                break
            except ValueError:
                continue
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        if df['date'].dt.tz is None:
            df['date'] = df['date'].dt.tz_localize(LOCAL_TZ)

        return df.dropna(subset=['date'])

    except Exception as e:
        print(f"Error loading historical data: {e}")
        return pd.DataFrame(columns=['date', 'product', 'types', 'price', 'predicted_price', 'discount', 'festival'])

def save_data_to_github(df):
    try:
        local_path = 'product_daily_prices_2022_2025_monthly_changes.csv'

        desired_columns = ['date', 'product', 'types', 'price', 'predicted_price', 'discount', 'festival']
        save_df = df.copy()

        for col in desired_columns:
            if col not in save_df.columns:
                if col == 'discount':
                    save_df['discount'] = 0.0
                elif col == 'festival':
                    save_df['festival'] = ''
                else:
                    save_df[col] = '' if col == 'product' else 0.0

        save_df = save_df[desired_columns]

        if 'date' in save_df.columns:
            save_df['date'] = pd.to_datetime(save_df['date']).dt.tz_localize(None)

            save_df = save_df.drop_duplicates(
                subset=['date', 'product', 'types'],
                keep='last'
            )

            save_df['date'] = save_df['date'].dt.strftime('%Y-%m-%d')

        numeric_cols = ['price', 'predicted_price', 'discount']
        for col in numeric_cols:
            if col in save_df.columns:
                save_df[col] = save_df[col].round(2)

        string_columns = ['product', 'types', 'festival']
        for col in string_columns:
            if col in save_df.columns:
                save_df[col] = save_df[col].astype(str)

        save_df.to_csv(local_path, index=False)
        print(f"Data saved locally to {local_path}")

        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            print("GitHub token not found. Data saved locally only.")
            return False

        g = Github(github_token)
        repo = g.get_repo("YYLIM0829/Price_Preiction")

        csv_content = save_df.to_csv(index=False)

        try:
            contents = repo.get_contents("product_daily_prices_2022_2025_monthly_changes.csv")
            existing_content = contents.decoded_content.decode('utf-8')
            if existing_content == csv_content:
                print("No changes detected, skipping GitHub update")
                return True

            update_response = repo.update_file(
                path=contents.path,
                message=f"Update price data",
                content=csv_content,
                sha=contents.sha
            )
            print("GitHub update successful:", update_response)
            return True
        except Exception as e:
            if "404" in str(e):
                create_response = repo.create_file(
                    path="product_daily_prices_2022_2025_monthly_changes.csv",
                    message=f"Create price data",
                    content=csv_content
                )
                print("GitHub create successful:", create_response)
                return True
            else:
                print("GitHub error:", str(e))
                return False

    except Exception as e:
        print(f"Failed to update GitHub: {str(e)}")
        print(f"Data saved locally at: {local_path}")
        return False

def prepare_data_for_prediction(historical_df, product_name, types, min_samples=30):
    required_cols = ['date', 'product', 'types', 'price']

    product_data = historical_df.copy()

    product_data['product_clean'] = product_data['product'].astype(str).str.lower().str.strip()
    product_name_clean = str(product_name).lower().strip()

    product_data['types_clean'] = product_data['types'].astype(str).str.lower().str.strip()
    types_clean = str(types).lower().strip()

    mask = (
        (product_data['product_clean'] == product_name_clean) &
        (product_data['types_clean'].str.contains(types_clean))
    )

    product_data = product_data[mask].copy()

    if product_data.empty:
        print(f"\nWarning: No data found for product '{product_name}' in types '{types}'")
        print("Available products in this types:")
        print(historical_df[historical_df['types'].str.contains(types_clean, case=False)]['product'].unique())
        return product_data

    product_data['date'] = pd.to_datetime(product_data['date'], errors='coerce')
    product_data = product_data.dropna(subset=['date'])
    product_data = product_data.sort_values('date')
    product_data = product_data.drop_duplicates(subset=['date'], keep='last')

    product_data = product_data.set_index('date').sort_index()

    all_dates = pd.date_range(
        start=product_data.index.min(),
        end=product_data.index.max()
    )
    product_data = product_data.reindex(all_dates)

    for col in ['price', 'predicted_price', 'discount']:
        product_data[col] = product_data[col].ffill().fillna(0)

    for col in ['product', 'types', 'festival']:
        product_data[col] = product_data[col].ffill().fillna('')

    product_data['day_of_week'] = product_data.index.dayofweek
    product_data['day_of_month'] = product_data.index.day
    product_data['month'] = product_data.index.month
    product_data['year'] = product_data.index.year

    if len(product_data) < min_samples:
            print(f"Warning: Only {len(product_data)} samples available (minimum {min_samples} required)")
            return pd.DataFrame()

    if len(product_data) < 2:
        return pd.DataFrame()

    return product_data

def calculate_metrics(y_true, y_pred):
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
    }
    return metrics

def knn_predict(product_data, days_ahead=1):
    X = product_data[['day_of_week', 'day_of_month', 'month', 'year']]
    y = product_data['price']

    X_train, X_test = X[:-days_ahead], X[-days_ahead:]
    y_train, y_test = y[:-days_ahead], y[-days_ahead:]

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)

    next_day_features = pd.DataFrame({
        'day_of_week': [(product_data.index[-1].dayofweek + days_ahead) % 7],
        'day_of_month': [(product_data.index[-1].day + days_ahead) % 31],
        'month': [product_data.index[-1].month],
        'year': [product_data.index[-1].year]
    })

    prediction = model.predict(next_day_features)[0]
    if len(y_test) > 0:
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
    else:
        metrics = None

    return prediction, metrics

def random_forest_predict(product_data, days_ahead=1):
    X = product_data[['day_of_week', 'day_of_month', 'month', 'year']]
    y = product_data['price']
    
    if len(X) < 10: 
        return None, None
    
    test_size = min(0.2, 5/len(X))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, 
                                max_depth=5,
                                min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    
    next_day_features = pd.DataFrame({
        'day_of_week': [(product_data.index[-1].dayofweek + days_ahead) % 7],
        'day_of_month': [(product_data.index[-1].day + days_ahead) % 31],
        'month': [product_data.index[-1].month],
        'year': [product_data.index[-1].year]
    })
    
    prediction = model.predict(next_day_features)[0]
    
    if len(y_test) > 0:
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
    else:
        metrics = None
    
    return prediction, metrics

def lstm_predict(product_data, days_ahead=1):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(product_data[['price']])

    sequence_length = 7
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length - days_ahead + 1):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length:i+sequence_length+days_ahead])

    X = np.array(X)
    y = np.array(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(days_ahead)
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=20, verbose=0)

    last_sequence = scaled_data[-sequence_length:]
    prediction = model.predict(last_sequence.reshape(1, sequence_length, 1))
    prediction = scaler.inverse_transform(prediction)[0][-1]

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, days_ahead))
        y_true = scaler.inverse_transform(y_test.reshape(-1, days_ahead))
        metrics = calculate_metrics(y_true, y_pred)
    else:
        metrics = None

    return prediction, metrics

def predict_product_price(product_name, types, historical_data, days_ahead=1):
    product_data = prepare_data_for_prediction(historical_data, product_name, types)

    if len(product_data) < 7:
        return None, {'knn': None, 'rf': None, 'lstm': None}

    knn_pred, knn_metrics = knn_predict(product_data, days_ahead)
    rf_pred, rf_metrics = random_forest_predict(product_data, days_ahead)
    lstm_pred, lstm_metrics = lstm_predict(product_data, days_ahead)

    target_date = datetime.now(LOCAL_TZ) + timedelta(days=days_ahead)
    festival, festival_date = is_festival_period(target_date)

    if festival:
        discount = get_festival_discount(types, festival)
        price_increase = get_festival_price_increase(types, festival)

        knn_pred *= (1 - discount) * (1 + price_increase)
        rf_pred *= (1 - discount) * (1 + price_increase)
        lstm_pred *= (1 - discount) * (1 + price_increase)

    models = {
        'knn': {'pred': knn_pred, 'metrics': knn_metrics},
        'rf': {'pred': rf_pred, 'metrics': rf_metrics},
        'lstm': {'pred': lstm_pred, 'metrics': lstm_metrics}
    }

    valid_models = {k: v for k, v in models.items() if v['metrics'] is not None}

    if valid_models:
        best_model = min(valid_models.items(), key=lambda x: x[1]['metrics']['mae'])
        return best_model[1]['pred'], models
    else:
        return lstm_pred, models

def update_firestore_predictions():
    try:
        if db is None:
            print("Firestore not initialized - skipping update")
            return False

        current_products = load_data_from_firestore()
        historical_data = load_historical_data()

        if current_products.empty:
            print("No products found in Firestore")
            return False

        today = datetime.now(LOCAL_TZ)

        print("Products loaded from Firestore:")
        print(current_products[['name', 'type', 'price']].to_string())

        new_rows = []
        for _, row in current_products.iterrows():
            last_updated = row.get('last_updated')
            if last_updated and last_updated.tzinfo is None:
                last_updated = last_updated.replace(tzinfo=LOCAL_TZ)

            festival, festival_date = is_festival_period(today)
            if festival:
                discount = get_festival_discount(row.get('type', 'Unknown'), festival)
                festival_name = festival
            else:
                discount = 0.0
                festival_name = ''

            new_rows.append({
                'date': today,
                'product': str(row.get('name', '')),
                'types': str(row.get('type', 'Unknown')),
                'price': float(row.get('price', 0)),
                'predicted_price': float(row.get('future_price', 0)),
                'discount': float(discount),
                'festival': festival_name
            })

        current_df = pd.DataFrame(new_rows)

        updated_data = pd.concat([historical_data, current_df], ignore_index=True)
        updated_data = updated_data.sort_values('date')

        success_count = 0
        for _, row in current_products.iterrows():
            try:
                predicted_price, _ = predict_product_price(
                    row['name'],
                    row.get('type', 'Unknown'),
                    updated_data
                )

                if predicted_price is not None:
                    docs = db.collection('products').where('name', '==', row['name']).limit(1).stream()
                    for doc in docs:
                        doc.reference.update({
                            'future_price': round(float(predicted_price), 2),
                            'last_updated': datetime.now(LOCAL_TZ),
                            'type': row.get('type', 'Unknown')
                        })
                        success_count += 1
                        print(f"Updated {row['name']} successfully")

            except Exception as e:
                print(f"Error updating {row['name']}: {str(e)}")

        print(f"Successfully updated {success_count}/{len(current_products)} products")

        if not save_data_to_github(updated_data):
            print("Warning: GitHub save failed")

        return success_count > 0

    except Exception as e:
        print(f"Critical error in update_firestore_predictions: {str(e)}")
        return False

def plot_prediction_comparison(product_name, types, historical_data):
    product_data = prepare_data_for_prediction(historical_data, product_name, types)
    
    product_data_2025 = product_data[product_data.index.year == 2025]

    if product_data.empty:
        print(f"No data available for {product_name} ({types})")
        return
        
    plt.figure(figsize=(14, 6))
    
    plt.plot(product_data_2025.index, product_data_2025['price'], 
            label='Actual Price', color='blue', alpha=0.7)
    
    if 'predicted_price' in product_data_2025.columns:
            plt.plot(product_data_2025.index, product_data_2025['predicted_price'],
                label='Previous Predictions', color='orange', alpha=0.7)
    
    plt.title(f'Price History for {product_name} ({types})')
    plt.xlabel('Date')
    plt.ylabel('Price (RM)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_admin_interface():
    historical_data = load_historical_data()

    required_columns = ['product', 'types', 'price']
    for col in required_columns:
        if col not in historical_data.columns:
            print(f"Warning: Missing required column '{col}' in historical data")
            return None

    if 'festival' not in historical_data.columns:
        historical_data['festival'] = ''

    products = historical_data['product'].unique()

    product_dropdown = Dropdown(
        options=products,
        description='Product:',
        disabled=False
    )

    model_dropdown = Dropdown(
        options=['knn', 'rf', 'lstm', 'all'],
        description='Model:',
        value='all'
    )

    date_picker = DatePicker(
        description='Target Date:',
        value=datetime.now(LOCAL_TZ).date() + timedelta(days=1)
    )

    output = Output()

    def on_parameter_change(change):
        with output:
            output.clear_output()

            product_name = product_dropdown.value
            model_type = model_dropdown.value
            target_date = date_picker.value

            if not product_name:
                return

            product_mask = historical_data['product'] == product_name
            if not any(product_mask):
                print(f"No data found for product: {product_name}")
                return

            types = historical_data[product_mask]['types'].iloc[0]
            
            plot_prediction_comparison(product_name, types, historical_data)
            
            days_ahead = (target_date - datetime.now(LOCAL_TZ).date()).days
            if days_ahead <= 0:
                print("Please select a future date")
                return

            predicted_price, model_results = predict_product_price(
                product_name, types, historical_data, days_ahead
            )

            if predicted_price is None:
                print("Could not generate prediction - insufficient historical data")
                return

            print(f"\nPredicted price for {target_date}: RM{predicted_price:.2f}\n")

            if model_type == 'all':
                print("Model Performance:")
                for model_name, result in model_results.items():
                    if result['metrics'] is not None:
                        print(f"\n{model_name.upper()}:")
                        for metric, value in result['metrics'].items():
                            print(f"{metric.upper()}: {value:.4f}")
                        print(f"Prediction: {result['pred']:.2f}")
                    else:
                        print(f"\n{model_name.upper()}: Not enough data for metrics")
                        print(f"Prediction: {result['pred']:.2f}")
            else:
                result = model_results[model_type]
                if result['metrics'] is not None:
                    print(f"\n{model_type.upper()} Metrics:")
                    for metric, value in result['metrics'].items():
                        print(f"{metric.upper()}: {value:.4f}")
                else:
                    print(f"{model_type.upper()}: Not enough data for metrics calculation")
                print(f"Prediction: {result['pred']:.2f}")

    product_dropdown.observe(on_parameter_change, names='value')
    model_dropdown.observe(on_parameter_change, names='value')
    date_picker.observe(on_parameter_change, names='value')

    display(product_dropdown, model_dropdown, date_picker, output)
    on_parameter_change(None)

if __name__ == "__main__":
    if db is not None:
        try:
            docs = db.collection('products').limit(1).stream()
            print("Firestore connection successful")
        except Exception as e:
            print("Firestore connection failed:", str(e))

    update_firestore_predictions()

    #create_admin_interface()
