import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from google.cloud import firestore
import pytz
import requests
import io
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import holidays
from ipywidgets import interact, Dropdown, DatePicker, Output
from IPython.display import display

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
        'duration': 7
    },
    'Mother\'s Day': {
        'discount': {'all': (0.05, 0.15)},
        'duration': 7
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

# Malaysian holidays for price adjustments
malaysia_holidays = holidays.Malaysia()

def is_festival_period(date):
    """Check if date falls within any festival period"""
    date_str = date.strftime('%Y-%m-%d')

    # Fixed date festivals
    festivals = {
        'Chinese New Year': [(1, 22), (1, 23)],  # Example dates for 2023
        'Labour Day': [(5, 1)],
        'Hari Raya': [(4, 21)],  # Example date
        'Father\'s Day': [(6, 18)],  # 3rd Sunday of June
        'Mother\'s Day': [(5, 14)],  # 2nd Sunday of May
        'Deepavali': [(10, 24)],  # Example date
        'Christmas': [(12, 25)],
        'National Day': [(8, 31)],
        '11.11 Sale': [(11, 11)],
        'Black Friday': [(11, 25)],  # 4th Friday of November
        'Year-End Sale': [(12, 15)]  # Starts Dec 15 for example
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

def get_festival_discount(category, festival):
    """Get discount range for a category during festival"""
    if festival not in FESTIVAL_DISCOUNTS:
        return 0

    festival_data = FESTIVAL_DISCOUNTS[festival]

    # Check category-specific discounts first
    if 'discount' in festival_data:
        for discount_category, discount_range in festival_data['discount'].items():
            if discount_category == 'all' or discount_category in category.lower():
                return np.random.uniform(*discount_range)

    return 0

def get_festival_price_increase(category, festival):
    """Get price increase range for high-demand categories during festival"""
    if festival not in FESTIVAL_DISCOUNTS:
        return 0

    festival_data = FESTIVAL_DISCOUNTS[festival]

    if 'price_increase' in festival_data:
        for inc_category, inc_range in festival_data['price_increase'].items():
            if inc_category in category.lower():
                return np.random.uniform(*inc_range)

    return 0

def load_data_from_firestore():
    """Load product data from Firestore"""
    products_ref = db.collection('products')
    docs = products_ref.stream()

    data = []
    for doc in docs:
        product = doc.to_dict()
        data.append({
            'name': product.get('name', ''),
            'type': product.get('type', ''),
            'price': product.get('price', 0),
            'future_price': product.get('future_price', 0),
            'last_updated': product.get('last_updated', datetime.now()).astimezone(LOCAL_TZ)
        })

    return pd.DataFrame(data)

def load_historical_data():
    """Load historical data with robust column handling"""
    try:
        response = requests.get(GITHUB_RAW_URL)
        response.raise_for_status()
        
        # Define expected columns and their default values
        expected_columns = {
            'date': None,
            'product': None,
            'category': None,
            'price': None,
            'predicted_price': 0.0,
            'discount': 0.0,
            'festival': ''
        }
        
        # Load the data
        df = pd.read_csv(io.StringIO(response.text))
        
        # Add missing columns with default values
        for col, default_val in expected_columns.items():
            if col not in df.columns:
                df[col] = default_val
        
        # Convert date column with multiple format attempts
        date_formats = ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d']
        for fmt in date_formats:
            try:
                df['date'] = pd.to_datetime(df['date'], format=fmt, errors='raise')
                break
            except ValueError:
                continue
        else:
            # If all formats fail, use flexible parsing
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['date'])
        return df
        
    except Exception as e:
        print(f"Error loading historical data: {e}")
        # Return empty DataFrame with all expected columns
        return pd.DataFrame(columns=expected_columns.keys())

def save_data_to_github(df):
    """Save updated data back to GitHub (you'll need to implement proper GitHub API calls)"""
    # This is a placeholder - in practice you'd use GitHub API or git commands
    # For Colab, you might want to save to Google Drive instead
    df.to_csv('product_prices_history.csv', index=False)
    print("Data saved locally - implement GitHub upload separately")

def prepare_data_for_prediction(historical_df, product_name, category):
    """Prepare data with more flexible matching"""
    # Create a copy to avoid SettingWithCopyWarning
    product_data = historical_df.copy()
    
    # Convert to lowercase and strip whitespace for more flexible matching
    product_data['product_clean'] = product_data['product'].astype(str).str.lower().str.strip()
    product_name_clean = str(product_name).lower().strip()
    
    product_data['category_clean'] = product_data['category'].astype(str).str.lower().str.strip()
    category_clean = str(category).lower().strip()
    
    # Filter for this product (more flexible matching)
    mask = (
        (product_data['product_clean'] == product_name_clean) & 
        (product_data['category_clean'].str.contains(category_clean))
    )
    
    product_data = product_data[mask].copy()
    
    if product_data.empty:
        print(f"\nWarning: No data found for product '{product_name}' in category '{category}'")
        print("Available products in this category:")
        print(historical_df[historical_df['category'].str.contains(category_clean, case=False)]['product'].unique())
        return product_data
    
    # Ensure proper datetime index
    product_data['date'] = pd.to_datetime(product_data['date'], errors='coerce')
    product_data = product_data.dropna(subset=['date'])
    product_data = product_data.set_index('date').sort_index()
    
    # Forward fill missing values
    all_dates = pd.date_range(
        start=product_data.index.min(),
        end=product_data.index.max()
    )
    product_data = product_data.reindex(all_dates)
    
    # Fill numeric columns
    for col in ['price', 'predicted_price', 'discount']:
        product_data[col] = product_data[col].ffill().fillna(0)
    
    # Fill string columns
    for col in ['product', 'category', 'festival']:
        product_data[col] = product_data[col].ffill().fillna('')
    
    # Feature engineering
    product_data['day_of_week'] = product_data.index.dayofweek
    product_data['day_of_month'] = product_data.index.day
    product_data['month'] = product_data.index.month
    product_data['year'] = product_data.index.year
    
    return product_data

def knn_predict(product_data, days_ahead=1):
    """KNN prediction model"""
    X = product_data[['day_of_week', 'day_of_month', 'month', 'year']]
    y = product_data['price']

    # Split data (last 'days_ahead' days for testing)
    X_train, X_test = X[:-days_ahead], X[-days_ahead:]
    y_train, y_test = y[:-days_ahead], y[-days_ahead:]

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)

    # Predict next day
    next_day_features = pd.DataFrame({
        'day_of_week': [(product_data.index[-1].dayofweek + days_ahead) % 7],
        'day_of_month': [(product_data.index[-1].day + days_ahead) % 31],
        'month': [product_data.index[-1].month],
        'year': [product_data.index[-1].year]
    })

    prediction = model.predict(next_day_features)[0]
    mae = mean_absolute_error(y_test, model.predict(X_test)) if len(y_test) > 0 else np.nan

    return prediction, mae

def random_forest_predict(product_data, days_ahead=1):
    """Random Forest prediction model"""
    X = product_data[['day_of_week', 'day_of_month', 'month', 'year']]
    y = product_data['price']

    # Split data
    X_train, X_test = X[:-days_ahead], X[-days_ahead:]
    y_train, y_test = y[:-days_ahead], y[-days_ahead:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict next day
    next_day_features = pd.DataFrame({
        'day_of_week': [(product_data.index[-1].dayofweek + days_ahead) % 7],
        'day_of_month': [(product_data.index[-1].day + days_ahead) % 31],
        'month': [product_data.index[-1].month],
        'year': [product_data.index[-1].year]
    })

    prediction = model.predict(next_day_features)[0]
    mae = mean_absolute_error(y_test, model.predict(X_test)) if len(y_test) > 0 else np.nan

    return prediction, mae

def lstm_predict(product_data, days_ahead=1):
    """LSTM prediction model"""
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(product_data[['price']])

    # Prepare sequences
    sequence_length = 7
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length - days_ahead + 1):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length:i+sequence_length+days_ahead])

    X = np.array(X)
    y = np.array(y)

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(days_ahead)
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=20, verbose=0)

    # Predict
    last_sequence = scaled_data[-sequence_length:]
    prediction = model.predict(last_sequence.reshape(1, sequence_length, 1))
    prediction = scaler.inverse_transform(prediction)[0][-1]

    # Calculate MAE if we have test data
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, days_ahead))
        y_true = scaler.inverse_transform(y_test.reshape(-1, days_ahead))
        mae = mean_absolute_error(y_true, y_pred)
    else:
        mae = np.nan

    return prediction, mae

def predict_product_price(product_name, category, historical_data, days_ahead=1):
    """Predict price using all models and return the best prediction"""
    product_data = prepare_data_for_prediction(historical_data, product_name, category)

    if len(product_data) < 7:  # Not enough data
        return None, {'knn': None, 'rf': None, 'lstm': None}

    # Get predictions from all models
    knn_pred, knn_mae = knn_predict(product_data, days_ahead)
    rf_pred, rf_mae = random_forest_predict(product_data, days_ahead)
    lstm_pred, lstm_mae = lstm_predict(product_data, days_ahead)

    # Adjust for festivals
    target_date = datetime.now(LOCAL_TZ) + timedelta(days=days_ahead)
    festival, festival_date = is_festival_period(target_date)

    if festival:
        discount = get_festival_discount(category, festival)
        price_increase = get_festival_price_increase(category, festival)

        knn_pred *= (1 - discount) * (1 + price_increase)
        rf_pred *= (1 - discount) * (1 + price_increase)
        lstm_pred *= (1 - discount) * (1 + price_increase)

    # Select model with lowest MAE (when available)
    models = {
        'knn': {'pred': knn_pred, 'mae': knn_mae},
        'rf': {'pred': rf_pred, 'mae': rf_mae},
        'lstm': {'pred': lstm_pred, 'mae': lstm_mae}
    }

    # Filter out models with no MAE (insufficient test data)
    valid_models = {k: v for k, v in models.items() if not np.isnan(v['mae'])}

    if valid_models:
        best_model = min(valid_models.items(), key=lambda x: x[1]['mae'])
        return best_model[1]['pred'], models
    else:
        # If no model has MAE (all predictions on training data), use LSTM as default
        return lstm_pred, models

def update_firestore_predictions():
    """Main function with improved column handling"""
    # Load current data
    current_products = load_data_from_firestore()
    historical_data = load_historical_data()
    
    # Validate and clean historical data
    if not historical_data.empty:
        # Ensure all required columns exist
        required_columns = [
            'date', 'product', 'category', 'price',
            'predicted_price', 'discount', 'festival'
        ]
        
        for col in required_columns:
            if col not in historical_data.columns:
                if col == 'discount':
                    historical_data[col] = 0.0
                elif col == 'festival':
                    historical_data[col] = ''
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Ensure date column is datetime and timezone-aware
        if not pd.api.types.is_datetime64_any_dtype(historical_data['date']):
            historical_data['date'] = pd.to_datetime(
                historical_data['date'],
                errors='coerce'
            )
        
        # Localize to the same timezone if not already aware
        if historical_data['date'].dt.tz is None:
            historical_data['date'] = historical_data['date'].dt.tz_localize(LOCAL_TZ)
        else:
            historical_data['date'] = historical_data['date'].dt.tz_convert(LOCAL_TZ)
            
        historical_data = historical_data.dropna(subset=['date'])
        
        # Ensure numeric columns are float
        for col in ['price', 'predicted_price', 'discount']:
            historical_data[col] = pd.to_numeric(
                historical_data[col],
                errors='coerce'
            ).fillna(0)
    
    # Prepare today's date with proper timezone
    today = datetime.now(LOCAL_TZ)
    
    # Update historical data with current prices
    new_rows = []
    for _, row in current_products.iterrows():
        new_rows.append({
            'date': today,  # This is already timezone-aware
            'product': str(row['name']),
            'category': str(row['type']),
            'price': float(row['price']),
            'predicted_price': float(row.get('future_price', 0)),
            'discount': 0.0,
            'festival': ''
        })
    
    # Combine old and new data
    updated_data = pd.concat([
        historical_data,
        pd.DataFrame(new_rows)
    ], ignore_index=True)
    
    # Ensure all dates are timezone-aware and in the same timezone
    if 'date' in updated_data and updated_data['date'].dt.tz is None:
        updated_data['date'] = updated_data['date'].dt.tz_localize(LOCAL_TZ)
    
    # Ensure proper sorting by date
    updated_data = updated_data.sort_values('date')
    
    # Predict next day's prices and update Firestore
    for _, row in current_products.iterrows():
        predicted_price, model_results = predict_product_price(
            row['name'], row['type'], updated_data
        )
        
        if predicted_price is not None:
            # Update Firestore
            product_ref = db.collection('products').where('name', '==', row['name']).limit(1).get()
            for doc in product_ref:
                doc.reference.update({
                    'future_price': float(predicted_price),
                    'last_updated': datetime.now(LOCAL_TZ)
                })
    
    # Save updated historical data
    save_data_to_github(updated_data)
    print("Price predictions updated successfully")

# Admin Interface
def create_admin_interface():
    """Create interactive admin interface for model evaluation"""
    historical_data = load_historical_data()
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

            category = historical_data[historical_data['product'] == product_name]['category'].iloc[0]
            product_data = prepare_data_for_prediction(historical_data, product_name, category)

            # Plot historical prices
            plt.figure(figsize=(12, 6))
            plt.plot(product_data.index, product_data['price'], label='Historical Prices')
            plt.title(f'Price History for {product_name}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()
            plt.show()

            # Get prediction
            days_ahead = (target_date - datetime.now(LOCAL_TZ).date()).days
            if days_ahead <= 0:
                print("Please select a future date")
                return

            predicted_price, model_results = predict_product_price(
                product_name, category, historical_data, days_ahead
            )

            print(f"Predicted price for {target_date}: RM{predicted_price:.2f}\n")

            if model_type == 'all':
                print("Model Performance:")
                for model_name, result in model_results.items():
                    if result['mae'] is not None:
                        print(f"{model_name.upper()}: MAE = {result['mae']:.2f}, Prediction = {result['pred']:.2f}")
                    else:
                        print(f"{model_name.upper()}: Not enough data for MAE, Prediction = {result['pred']:.2f}")
            else:
                result = model_results[model_type]
                if result['mae'] is not None:
                    print(f"{model_type.upper()} MAE: {result['mae']:.2f}")
                else:
                    print(f"{model_type.upper()}: Not enough data for MAE calculation")

    product_dropdown.observe(on_parameter_change, names='value')
    model_dropdown.observe(on_parameter_change, names='value')
    date_picker.observe(on_parameter_change, names='value')

    display(product_dropdown, model_dropdown, date_picker, output)
    on_parameter_change(None)  # Initial call

# Main execution
if __name__ == "__main__":
    # For daily runs in Colab
    update_firestore_predictions()

    # Uncomment to enable admin interface when running interactively
    # create_admin_interface()
