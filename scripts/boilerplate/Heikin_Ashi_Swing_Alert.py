'''
-This script contains a sample multi-variate crossover strategy with the MACD, Stochastic, and Hiekin Ashi Bar Color (e.g. Lag1 Red Bar and Cur Green Bar)
-This script uses Heikin Ashi Bars on a 1-day timeframe
-There is no exit strategy or stop loss, purely used for a buy or sell alert only.
-Recommend you make a separate yml config file (copy & paste) for Alert scripts.
-Review Bot_Tutorials.md file
-Review README.md
-*IMPORTANT* as with all scripts, make sure date & time parameters are set properly for the script or scripts your running. If not you will pull unnessary data or your code may not run how it should.
-*IMPORTANT* Never ever ever run a live strategy without forward testing in a paper account to find efficacy. Monitor closely and make sure parameters and dates are set properly.
'''

strat_name = 'Heiken_Ashi_Swing_Alert' # Strat Name
bot_type = 'PAPER' # paper or live
config_file_path = ""
# "C:/Users/.../..../..../boilerplate_config.yml"

#####################################Section 1: Config, Modules, and Imports############################

# Modules
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import alpaca_trade_api as tradeapi
import datetime as dt
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from pytz import timezone
import time
import yaml


# Load YAML configuration
with open(config_file_path, "r") as file:
    config = yaml.safe_load(file)

# Access variables
API_KEY = config['API_KEY']
SECRET_KEY = config["SECRET_KEY"]
api_base_url = config["api_base_url"]
symbols = config["symbols"] #remove this if adding locally, add symbols = [''] below.
position_size = config["position_size"]
start_time = config["start_time"]
end_time = config["end_time"]

# Client/Acc Info
print("ACC INFO: ")
print("-----------------------------------------")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True) #switch to paper=False for Live. New base URL for live.

account = trading_client.get_account()
for property_name, value in account:
    print(f"\"{property_name}\":{value}")
   
hist_data_client = StockHistoricalDataClient(API_KEY,SECRET_KEY) #key validation

api = tradeapi.REST(API_KEY, SECRET_KEY, api_base_url, api_version='v2')

# symbol list length
num_of_symbols = len(symbols)
print("-----------------------------------------")
print(f"The number of tickers in this list are {num_of_symbols}")

# Daily Buying power
daytrading_buying_power = int(round(float(account.daytrading_buying_power)))
print(f"Position sizes are currently set to: ${position_size}")
print(f"Day Trading Buying Power: ${daytrading_buying_power}")

# Max Trade Attempts
max_trade_attempts = int(round(float(daytrading_buying_power/position_size)))

###################################Section 2: Global/Key Variable Settings#####################################

# add symbols here if you do not want to store them via YML config file
# symbols = ['NVDA','TSLA','MSFT']

'''
TIME VALUES (UPDATE BASED ON STRATEGY)
'''

# Timeframe constraints ("Minute","Hour","Day") ... See custom_timeframe_example file or alpaca docs if looking to implement 5m or custom bar timeframes.
# REMEMBER MUST HAVE ALGO PLUS SUBSCRIPTION TO ACCESS BAR DATA WITHIN 15m of real-time!
timeframe = "Day"

# Start/End Time constraints (Start Time 9,30 = market open9:30am est)
hour_start_time = 9
minute_start_time = 33
hour_end_time = 18
minute_end_time = 15

# polling interval is set for 1 second. Script updates automatically every 60 seconds. See Main func. Recommendation is 1 -30 seconds depending when during the candlestick you want to enter.
polling_interval = 1

'''
INDICATOR VALUES (UPDATE BASED ON STRATEGY)

Only a few indicator values have been hard coded as global variables, you can add as many as you would like :)
'''
# MACD 
macd_fast_period_val = 8
macd_slow_period_val = 18
macd_signal_period_val = 2
# EMA
ema_fast_period_val = 3
ema_mod_period_val =  6
ema_slow_period_val = 200
ema_20_period_val = 20
# SMA
sma_fast_period_val = 20
sma_slow_period_val = 50
# BB
bb_dev_factor_intense_val = 3.0
bb_dev_factor_tame_val = 1.0
# PSAR
psar_acceleration_dev_factor = 0.02
psar_maximum_dev_factor = 0.2
# Stochastic
stoch_slowK_period_val = 3
stoch_slowD_period_val = 3
stoch_fastK_period_val = 14

'''
Take Profit / Stop Loss Params
'''
# This is a buy or sell alert strategy with no predetermined stop loss or exit conditions.

###################################Section 3: Key Functions#####################################

# Fetching OHLC Bar Data from Alpaca API
def get_stock_bars_data(api_key: str, secret_key: str, tickers: list, start_date: str, end_date: str):
    hist_data_client = StockHistoricalDataClient(api_key, secret_key)  # key validation

    # Use the time constraint variable `timeframe` with getattr
    timeframe_attr = getattr(TimeFrame, timeframe)

    request_params = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=timeframe_attr,
        start=dt.datetime.strptime(start_date, "%Y-%m-%d"),
        end=dt.datetime.strptime(end_date, "%Y-%m-%d")
    )

    ticker_bars = hist_data_client.get_stock_bars(request_params)  # validation call
    return ticker_bars.df

# Converting OHLC Bar Data to Heiken Ashi Bar Data (Essentially the "average" bar price, with an associated bar color, red or green)
# Extremely useful for developing crossover heiken ashi strategies
def calculate_heiken_ashi(df, open_col="open", high_col="high", low_col="low", close_col="close"):
    """
    Calculate the Heiken Ashi bars for a pandas DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        open_col (str): Name of the column in the DataFrame that contains the open prices.
        high_col (str): Name of the column in the DataFrame that contains the high prices.
        low_col (str): Name of the column in the DataFrame that contains the low prices.
        close_col (str): Name of the column in the DataFrame that contains the close prices.

    Returns:
        pandas.DataFrame: DataFrame containing the Heiken Ashi values (HA_open, HA_high, HA_low, HA_close, color).
    """
    # Calculate HA_close as the average of open, high, low, and close
    ha_close = (df[open_col] + df[high_col] + df[low_col] + df[close_col]) / 4
    
    # Initialize HA_open with the first element based on the first available open and close prices
    ha_open = [(df[open_col].iloc[0] + df[close_col].iloc[0]) / 2]
    
    # Iterate through the DataFrame from the second row onward to calculate subsequent HA_open values
    for i in range(1, len(df)):
        new_open = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2
        ha_open.append(new_open)

    # HA_high is the maximum of high, open, and close columns
    ha_high = df[[high_col, open_col, close_col]].max(axis=1)
    
    # HA_low is the minimum of low, open, and close columns
    ha_low = df[[low_col, open_col, close_col]].min(axis=1)

    # Determine the color of the Heiken Ashi bars based on the relationship between HA_close and HA_open
    color = ["green" if ha_close.iloc[i] > ha_open[i] else "red" for i in range(len(ha_close))]

    # Compile the calculated values into a new DataFrame
    heiken_ashi_df = pd.DataFrame({
        "HA_open": ha_open,
        "HA_high": ha_high,
        "HA_low": ha_low,
        "HA_close": ha_close,
        "color": color
    }, index=df.index)
    
    return heiken_ashi_df


# TA-Lib MACD func
def calculate_macd(df, column="HA_close", fast_period=macd_fast_period_val, slow_period=macd_slow_period_val, signal_period=macd_signal_period_val):
    """Calculate the MACD (Moving Average Convergence Divergence) using TALIB on a pandas DataFrame.
  
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        fast_period (int): Number of periods to consider for the fast moving average. Default is 12.
        slow_period (int): Number of periods to consider for the slow moving average. Default is 26.
        signal_period (int): Number of periods to consider for the signal line. Default is 9.
  
    Returns:
        pandas.DataFrame: DataFrame containing the MACD values (macd, signal, histogram).
    """
    close_prices = df[column].astype(float).values
    macd, signal, histogram = talib.MACD(close_prices, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
  
    macd_df = pd.DataFrame({
        "macd": macd,
        "signal": signal,
        "histogram": histogram
    }, index=df.index)
  
    return macd_df


def calculate_rsi(df, column="HA_close", period=14):
    """Calculate the Relative Strength Index (RSI) using TALIB on a pandas DataFrame.
  
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        period (int): Number of periods to consider for the RSI calculation. Default is 14.
  
    Returns:
        pandas.Series: Series containing the RSI values.
    """
    close_prices = df[column].astype(float).values
    rsi = talib.RSI(close_prices, timeperiod=period)
    rsi_series = pd.Series(rsi, index=df.index).rename("RSI")
    return rsi_series


def calculate_apo(df, column="HA_close", fast_period=12, slow_period=26):
    """Calculate the APO (Absolute Price Oscillator) using TALIB on a pandas DataFrame.
  
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        fast_period (int): Number of periods to consider for the fast moving average. Default is 12.
        slow_period (int): Number of periods to consider for the slow moving average. Default is 26.
  
    Returns:
        pandas.DataFrame: DataFrame containing the APO values.
    """
    close_prices = df[column].astype(float).values
    apo = talib.APO(close_prices, fastperiod=fast_period, slowperiod=slow_period)

    apo_df = pd.DataFrame({
        "apo": apo
    }, index=df.index)

    return apo_df


def calculate_atr(df, column="HA_close", atr_period=14, multiplier=0.2):
    """Calculate the ATR (Average True Range) using TALIB on a pandas DataFrame.
  
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        atr_period (int): Number of periods to consider for the ATR calculation. Default is 14.
        multiplier (float): Factor to multiply the ATR value. Default is 1 (no multiplier).
  
    Returns:
        pandas.DataFrame: DataFrame containing the multiplied ATR values.
    """
    high_prices = df['high'].astype(float).values
    low_prices = df['low'].astype(float).values
    close_prices = df[column].astype(float).values

    atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)

    atr_df = pd.DataFrame({
        "atr": atr_values * multiplier,
    }, index=df.index)
  
    return atr_df


def calculate_parabolic_SAR(df, column="HA_close", acceleration=psar_acceleration_dev_factor, maximum=psar_maximum_dev_factor):
    """Calculate the Parabolic SAR using TALIB on a pandas DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        acceleration (float): The acceleration factor for the Parabolic SAR. Default is 0.02.
        maximum (float): The maximum value for the acceleration factor. Default is 0.2.

    Returns:
        pandas.DataFrame: DataFrame containing the Parabolic SAR values.
    """
    high_prices = df['high'].astype(float).values
    low_prices = df['low'].astype(float).values
    
    parabolic_sar = talib.SAR(high_prices, low_prices, acceleration=acceleration, maximum=maximum)
    
    sar_df = pd.DataFrame({
        "parabolic_sar": parabolic_sar
    }, index=df.index)
    
    return sar_df


def calculate_stochastic(df, high_col="HA_high", low_col="HA_low", close_col="HA_close", fastk_period=stoch_fastK_period_val, slowk_period=stoch_slowK_period_val, slowd_period=stoch_slowK_period_val):
    """Calculate the Stochastic oscillator using TALIB on a pandas DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        high_col (str): Name of the column in the DataFrame that contains the high prices. Default is "high".
        low_col (str): Name of the column in the DataFrame that contains the low prices. Default is "low".
        close_col (str): Name of the column in the DataFrame that contains the closing prices. Default is "close".
        fastk_period (int): Number of periods to use for the Fast %K calculation. Default is 14.
        slowk_period (int): Number of periods to use for the Slow %K calculation. Default is 3.
        slowd_period (int): Number of periods to use for the Slow %D calculation. Default is 3.

    Returns:
        pandas.DataFrame: DataFrame containing the Stochastic oscillator values.
    """
    high_prices = df[high_col].astype(float).values
    low_prices = df[low_col].astype(float).values
    close_prices = df[close_col].astype(float).values

    slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=fastk_period,
                               slowk_period=slowk_period, slowd_period=slowd_period)

    stochastic_df = pd.DataFrame({
        "SlowK": slowk,
        "SlowD": slowd
    }, index=df.index)

    return stochastic_df


def calculate_bollinger_bands_intense(df, column="HA_close", period=20, dev_factor=bb_dev_factor_intense_val):
    """Calculate Bollinger Bands using TALIB on a pandas DataFrame.
  
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        period (int): Number of periods to consider for the moving average. Default is 20.
        dev_factor (int): Number of standard deviations to consider for the bands. Default is 2.
  
    Returns:
        pandas.DataFrame: DataFrame containing the Bollinger Bands (upper_band, middle_band, lower_band).
    """
    close_prices = df[column].astype(float).values
    upper_band, middle_band, lower_band = talib.BBANDS(close_prices, timeperiod=period, nbdevup=dev_factor, nbdevdn=dev_factor)
  
    bollinger_df = pd.DataFrame({
        "upper_band": upper_band,
        "middle_band": middle_band,
        "lower_band": lower_band
    }, index=df.index)
  
    return bollinger_df


def calculate_bollinger_bands_tame(df, column="HA_close", period=20, dev_factor=bb_dev_factor_tame_val):
    """Compute modified Bollinger Bands on a pandas DataFrame.

    Args:
        data_frame (pandas.DataFrame): DataFrame with price data.
        price_column (str): Column name in the DataFrame with price values. Default is "modified_close".
        span (int): Time frame to consider for the moving average. Default is 25.
        std_factor (int): Standard deviation factor for the bands. Default is a custom value.

    Returns:
        pandas.DataFrame: DataFrame with the Bollinger Bands (upper_b, middle_b, lower_b).
    """
    price_values = df[column].astype(float).values
    upper_b, middle_b, lower_b = talib.BBANDS(price_values, timeperiod=period, nbdevup=dev_factor, nbdevdn=dev_factor)

    bbands_df = pd.DataFrame({
        "upper_b": upper_b,
        "middle_b": middle_b,
        "lower_b": lower_b
    }, index=df.index)

    return bbands_df


def calculate_sma_fast(df, column="HA_close", period=sma_fast_period_val):
    """Calculate the Simple Moving Average (SMA) using TALIB on a pandas DataFrame.
  
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        period (int): Number of periods to consider for the SMA calculation. Default is 20.
  
    Returns:
        pandas.Series: Series containing the SMA values.
    """
    close_prices = df[column].astype(float).values
    sma_fast = talib.SMA(close_prices, timeperiod=sma_fast_period_val)
    sma_series_fast = pd.Series(sma_fast, index=df.index).rename("SMA_Fast")
    return sma_series_fast


def calculate_sma_slow(df, column="HA_close", period=sma_slow_period_val):
    """Calculate the Simple Moving Average (SMA) using TALIB on a pandas DataFrame.
  
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        period (int): Number of periods to consider for the SMA calculation. Default is 20.
  
    Returns:
        pandas.Series: Series containing the SMA values.
    """
    close_prices = df[column].astype(float).values
    sma_slow = talib.SMA(close_prices, timeperiod=sma_slow_period_val)
    sma_series_slow = pd.Series(sma_slow, index=df.index).rename("SMA_Slow")
    return sma_series_slow


def calculate_ema_fast(df, column="HA_close", period=ema_fast_period_val):
    """Calculate the Exponential Moving Average (EMA) using TALIB on a pandas DataFrame.
  
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        period (int): Number of periods to consider for the EMA calculation. Default is 20.
  
    Returns:
        pandas.Series: Series containing the EMA values.
    """
    close_prices = df[column].astype(float).values
    ema_fast = talib.EMA(close_prices, timeperiod=ema_fast_period_val)
    ema_series_fast = pd.Series(ema_fast, index=df.index).rename("EMA_Fast")
    return ema_series_fast


def calculate_ema_mod(df, column="HA_close", period=ema_mod_period_val):
    """Calculate the Exponential Moving Average (EMA) using TALIB on a pandas DataFrame.
  
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        period (int): Number of periods to consider for the EMA calculation. Default is 20.
  
    Returns:
        pandas.Series: Series containing the EMA values.
    """
    close_prices = df[column].astype(float).values
    ema_mod = talib.EMA(close_prices, timeperiod=ema_mod_period_val)
    ema_series_mod = pd.Series(ema_mod, index=df.index).rename("EMA_Mod")
    return ema_series_mod


def calculate_ema_slow(df, column="HA_close", period=ema_slow_period_val):
    """Calculate the Exponential Moving Average (EMA) using TALIB on a pandas DataFrame.
  
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        period (int): Number of periods to consider for the EMA calculation. Default is 20.
  
    Returns:
        pandas.Series: Series containing the EMA values.
    """
    close_prices = df[column].astype(float).values
    ema_slow = talib.EMA(close_prices, timeperiod=ema_slow_period_val)
    ema_series_slow = pd.Series(ema_slow, index=df.index).rename("EMA_Slow")
    return ema_series_slow


def calculate_ema_20(df, column="HA_close", period=ema_20_period_val):
    """Calculate the Exponential Moving Average (EMA) using TALIB on a pandas DataFrame.
  
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        period (int): Number of periods to consider for the EMA calculation. Default is 20.
  
    Returns:
        pandas.Series: Series containing the EMA values.
    """
    close_prices = df[column].astype(float).values
    ema_20 = talib.EMA(close_prices, timeperiod=ema_slow_period_val)
    ema_series_20 = pd.Series(ema_20, index=df.index).rename("EMA_20")
    return ema_series_20


def calculate_vwap_drawdown(df, column="HA_close", vwap_column="vwap", drawdown_threshold=30.0):
    """
    Calculate the drawdown percentage from the VWAP price on a pandas DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        vwap_column (str): Name of the column in the DataFrame that contains the VWAP values. Default is "vwap".
        drawdown_threshold (float): Percentage threshold for drawdown. Default is 5.0.

    Returns:
        pandas.DataFrame: DataFrame containing the drawdown percentage values and a signal column indicating if the drawdown is below the threshold.
    """
    close_prices = df[column].astype(float).values
    vwap_prices = df[vwap_column].astype(float).values

    # Calculate drawdown percentage
    drawdown = ((vwap_prices - close_prices) / vwap_prices) * 100

    # Create a DataFrame with drawdown values and threshold signals
    drawdown_df = pd.DataFrame({
        "drawdown": drawdown,
        "below_threshold": drawdown > drawdown_threshold  # True if drawdown exceeds threshold
    }, index=df.index)

    return drawdown_df


#############################Rolling Average Functions#############################


def calculate_roc_and_avg(df, column="close", period=14):
    """
    Calculate the Rate of Change (ROC) and its rolling average using TA-LIB on a pandas DataFrame.
  
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "close".
        period (int): Number of periods to consider for the ROC. Default is 14.
  
    Returns:
        pandas.DataFrame: DataFrame containing the ROC values and their rolling average.
    """
    close_prices = df[column].astype(float).values
    roc_values = talib.ROC(close_prices, timeperiod=period)
    avg_roc_values = pd.Series(roc_values).rolling(window=period).mean().values

    roc_series = pd.Series(roc_values, name="ROC", index=df.index)
    avg_roc_series = pd.Series(avg_roc_values, name="avg_ROC", index=df.index)

    roc_result_df = pd.concat([roc_series, avg_roc_series], axis=1)
  
    return roc_result_df


def calculate_rolling_average_volume(df, column="volume", period=14):
    """
    Calculate the rolling average volume for the specified number of periods.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the volume data.
        column (str): Name of the column in the DataFrame that contains the volumes. Default is "volume".
        period (int): Number of periods to consider for the rolling average volume. Default is 14.
    
    Returns:
        pandas.Series: Series containing the rolling average volume values.
    """
    volumes = df[column]
    rolling_avg_volume_series = volumes.rolling(window=period).mean()
    
    return rolling_avg_volume_series

def calculate_rolling_average_low_price(df, column="HA_low", period=3):
    """
    Calculate the rolling average of the low prices for the specified number of periods.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the price values. Default is "low".
        period (int): Number of periods to consider for the rolling average of the low prices. Default is 14.
    
    Returns:
        pandas.Series: Series containing the rolling average of the low prices values.
    """
    prices = df[column]
    rolling_low_series = prices.rolling(window=period).min()
    rolling_avg_low_price_series = rolling_low_series.rolling(window=period).mean()
    
    return rolling_avg_low_price_series

def calculate_rolling_average_high_price(df, column="HA_high", period=3):
    """
    Calculate the rolling average of the high prices for the specified number of periods.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the price values. Default is "high".
        period (int): Number of periods to consider for the rolling average of the high prices. Default is 14.
    
    Returns:
        pandas.Series: Series containing the rolling average of the high prices values.
    """
    prices = df[column]
    rolling_high_series = prices.rolling(window=period).max()
    rolling_avg_high_price_series = rolling_high_series.rolling(window=period).mean()
    
    return rolling_avg_high_price_series


#############################Supportive Functions#############################

def check_order_status(api, order_id):
    """
    Checks the status of an order until it is either filled or canceled.
    """
    while True:
        order_info = api.get_order(order_id)
        if order_info.status in ['filled', 'canceled','pending']:
            return order_info
        time.sleep(1)  # Check every second
           
###################################Section 4: Test DataFrame Config####################################
# Feature Engineering
# Concat all indicator values into one DF = hist_data_raw()
#----------------------------------------------#
hist_data_raw = get_stock_bars_data(api_key=API_KEY,secret_key=SECRET_KEY,tickers=symbols,start_date=start_time,end_date=end_time)
heiken_ashi_df = calculate_heiken_ashi(df=hist_data_raw, open_col='open', high_col='high', low_col='low', close_col='close')
hist_data_raw['HA_open'] = heiken_ashi_df['HA_open']
hist_data_raw['HA_high'] = heiken_ashi_df['HA_high']
hist_data_raw['HA_low'] = heiken_ashi_df['HA_low']
hist_data_raw['HA_close'] = heiken_ashi_df['HA_close']
hist_data_raw['color'] = heiken_ashi_df['color']
#MACD concat
macd_df = calculate_macd(df=hist_data_raw,column='HA_close',fast_period=12, slow_period=26, signal_period=7)
hist_data_raw['macd'] = macd_df['macd']
hist_data_raw['signal'] = macd_df['signal']
hist_data_raw['histogram'] = macd_df['histogram']
#RSI concat
rsi_series = calculate_rsi(df=hist_data_raw,column='HA_close',period=14)
hist_data_raw['RSI'] = rsi_series
#ATR concat
atr_df = calculate_atr(df=hist_data_raw, column='HA_close',atr_period=14,multiplier=0.2)
hist_data_raw['atr'] = atr_df['atr']
#Parabolic SAR concat
sar_df = calculate_parabolic_SAR(df=hist_data_raw, column='HA_close',acceleration=0.02, maximum=0.2)
hist_data_raw['parabolic_sar'] = sar_df['parabolic_sar']
#Stochastic concat
stochastic_df = calculate_stochastic(df=hist_data_raw, high_col='HA_high',low_col='HA_low',close_col='HA_close',fastk_period=14,slowk_period=3, slowd_period=3)
hist_data_raw['SlowK'] = stochastic_df['SlowK']
hist_data_raw['SlowD'] = stochastic_df['SlowD']
#Bollinger concat
bollinger_df = calculate_bollinger_bands_intense(df=hist_data_raw, column='HA_close', period=20, dev_factor=bb_dev_factor_intense_val)
hist_data_raw['upper_band'] = bollinger_df['upper_band']
hist_data_raw['middle_band'] = bollinger_df['middle_band']
hist_data_raw['lower_band'] = bollinger_df['lower_band']
bbands_df = calculate_bollinger_bands_tame(df=hist_data_raw, column="HA_close", period=20, dev_factor=bb_dev_factor_tame_val)
hist_data_raw['upper_b'] = bbands_df['upper_b']
hist_data_raw['lower_b'] = bbands_df['lower_b']
#SMA_Fast Concat
sma_series = calculate_sma_fast(df=hist_data_raw, column='HA_close',period='10')
hist_data_raw['SMA_Fast'] = sma_series
#EMA_Fast Concat
ema_series_fast = calculate_ema_fast(df=hist_data_raw, column='HA_close',period='5')
hist_data_raw['EMA_Fast'] = ema_series_fast
#EMA_Slow Concat
ema_series_slow = calculate_ema_slow(df=hist_data_raw, column='HA_close',period='20')
hist_data_raw['EMA_Slow'] = ema_series_slow
#rolling avg volume concat
rolling_avg_volume_series = calculate_rolling_average_volume(df=hist_data_raw, column='volume', period=14)
hist_data_raw['rolling_avg_volume'] = rolling_avg_volume_series
#rolling avg Max/Min Price concat
rolling_avg_high_price_series = calculate_rolling_average_high_price(df=hist_data_raw, column='high', period=14)
hist_data_raw['rolling_avg_high_price'] = rolling_avg_high_price_series
rolling_avg_low_price_series = calculate_rolling_average_low_price(df=hist_data_raw, column='low', period=14)
hist_data_raw['rolling_avg_low_price'] = rolling_avg_low_price_series
#concat finished
hist_data_raw.dropna(inplace=True)
print(hist_data_raw)
print(hist_data_raw.index.nlevels > 1)

###################################SEQUENCING#################################

print("\n")
print(f'The start date for this code is {start_time}')
print(f'The end date for this code is {end_time}')
print("--------------------------")
print("\n")
print(f"Position sizes are currently set to: ${position_size}")
print(f"Day Trading Buying Power: ${daytrading_buying_power}")
print("--------------------------")
print("\n")
print("EXECUTE PROGRAM: BOT BRIE")
print("--------------------------")
print("    /\_/\  (  ")
print("   ( ^.^ )__)  ")
print("    )/q"" (   ")
print("   ( | | )   ")
print("  (__d b__)  ")
print("\n")
print("_                            ")
print(":`.            .--._         ")
print("`.`-.        /  ',--""'''''    ")
print("  `. ``~-._.'_.'/            ")
print("    `~-._ .` `~;             ")
print("        ;.    /              ")
print("       /     /               ")
print("  ,_.-';_,.'`                ")
print("    `-;`/                    ")
print("     ,'`                     ")
print("\n")

###################################Section 5: Main Function#####################################

def main():
    est = timezone('US/Eastern')
    combined_trade_count = 0
    
    while True:
        datezone_est = datetime.now(est)
        
        # The script continuously checks the current time against the defined market start and end times in the Eastern timezone. Trading operations only occur within these hours.
        market_start_time = datezone_est.replace(hour=hour_start_time, minute=minute_start_time, second=0, microsecond=0)
        market_end_time = datezone_est.replace(hour=hour_end_time, minute=minute_end_time, second=0, microsecond=0)

        if datezone_est >= market_start_time and datezone_est <= market_end_time:
            print("starting iteration at {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            print("\n")
            print("------------New Iteration-------------")
            # Calls the Alpaca API to fetch historical stock data. Appends all data to `stock_data` for real-time processing.
            stock_data = get_stock_bars_data(api_key=API_KEY, secret_key=SECRET_KEY, tickers=symbols, start_date=start_time, end_date=end_time)
            # Begin appending indicator data to stock_data df
            # Heiken Ashi Bar Conversion
            heiken_ashi_df = calculate_heiken_ashi(df=stock_data, open_col='open', high_col='high', low_col='low', close_col='close')
            stock_data['HA_open'] = heiken_ashi_df['HA_open']
            stock_data['HA_high'] = heiken_ashi_df['HA_high']
            stock_data['HA_low'] = heiken_ashi_df['HA_low']
            stock_data['HA_close'] = heiken_ashi_df['HA_close']
            stock_data['color'] = heiken_ashi_df['color']
            #
            rsi_series = calculate_rsi(df=stock_data, column='HA_close', period=14)
            stock_data['RSI'] = rsi_series
            #
            ema_series_fast = calculate_ema_fast(df=stock_data, column='HA_close', period=ema_fast_period_val)
            stock_data['EMA_Fast'] = ema_series_fast
            #
            ema_series_mod = calculate_ema_mod(df=stock_data, column="HA_close", period=ema_mod_period_val)
            stock_data['EMA_Mod'] = ema_series_mod
            #
            ema_series_slow = calculate_ema_slow(df=stock_data, column='HA_close', period=ema_slow_period_val)
            stock_data['EMA_Slow'] = ema_series_slow
            #
            ema_series_20 = calculate_ema_20(df=stock_data, column='HA_close', period=ema_20_period_val)
            stock_data['EMA_20'] = ema_series_20
            #
            sma_series_fast = calculate_sma_fast(df=stock_data, column='HA_close', period=sma_fast_period_val)
            stock_data['SMA_Fast'] = sma_series_fast
            #
            sma_series_slow = calculate_sma_slow(df=stock_data, column='HA_close', period=sma_slow_period_val)
            stock_data['SMA_Slow'] = sma_series_slow
            #
            macd_df = calculate_macd(df=stock_data,column='HA_close', fast_period=macd_fast_period_val, slow_period=macd_slow_period_val, signal_period=macd_signal_period_val)
            stock_data['macd'] = macd_df['macd']
            stock_data['signal'] = macd_df['signal']
            stock_data['histogram'] = macd_df['histogram']
            #
            stochastic_df = calculate_stochastic(df=stock_data, high_col='HA_high', low_col='HA_low', close_col='HA_close', fastk_period=stoch_fastK_period_val, slowk_period=stoch_slowK_period_val, slowd_period=stoch_slowD_period_val)
            stock_data['SlowK'] = stochastic_df['SlowK']
            stock_data['SlowD'] = stochastic_df['SlowD']
            #
            sar_df = calculate_parabolic_SAR(df=stock_data, column='HA_close', acceleration=psar_acceleration_dev_factor, maximum=psar_maximum_dev_factor)
            stock_data['parabolic_sar'] = sar_df['parabolic_sar']
            #
            bollinger_df = calculate_bollinger_bands_intense(df=stock_data, column="HA_close", period=20, dev_factor=bb_dev_factor_intense_val)
            stock_data['upper_band'] = bollinger_df['upper_band']
            stock_data['lower_band'] = bollinger_df['lower_band']
            bbands_df = calculate_bollinger_bands_tame(df=stock_data, column="HA_close", period=20, dev_factor=bb_dev_factor_tame_val)
            stock_data['upper_b'] = bbands_df['upper_b']
            stock_data['lower_b'] = bbands_df['lower_b']
            #
            rolling_avg_volume_series = calculate_rolling_average_volume(df=stock_data, column="volume", period=14)
            stock_data['rolling_avg_volume'] = rolling_avg_volume_series
            #
            rolling_avg_high_price_series = calculate_rolling_average_high_price(df=stock_data, column='HA_high', period=14)
            stock_data['rolling_avg_high_price'] = rolling_avg_high_price_series
            #
            rolling_avg_low_price_series = calculate_rolling_average_low_price(df=stock_data, column='HA_low', period=14)
            stock_data['rolling_avg_low_price'] = rolling_avg_low_price_series
            #Concat Completed for indicators to main DF (stock_data)
            #stock_data = stock_data.reset_index()
            #print(stock_data.index.nlevels > 1)
            stock_data.dropna(inplace=True)
            print(stock_data)
        
            positions = api.list_positions()
          
            for symbol in symbols:
                #print(stock_data.index)
                symbol_data = stock_data.loc[symbol] # Get data for the current symbol (symbol data is where ALL of our data will be stored)
                cur_period_data = symbol_data.iloc[-1] # Getting most recent row of data (Creating the timeseries data by time increment relative to real-time, these will become our bar increments)
                lag1_period_data = symbol_data.iloc[-2] # Delayed by 1 periods
                lag2_period_data = symbol_data.iloc[-3] # Delayed by 2 periods
                lag3_period_data = symbol_data.iloc[-4] # Delayed by 3 periods
                lag4_period_data = symbol_data.iloc[-5] # Delayed by 4 periods
                lag5_period_data = symbol_data.iloc[-6] # Delayed by 5 periods
                lag7_period_data = symbol_data.iloc[-8] #Delayed by 7 periods
                # CLOSING bar/lag bar data
                cur_bar_data = cur_period_data['HA_close'] # Get the last row (latest period)
                lag1_bar_data = lag1_period_data['HA_close']
                lag2_bar_data = lag2_period_data['HA_close']
                lag3_bar_data = lag3_period_data['HA_close']
                # HA Color
                cur_HA_color = cur_period_data['color']
                lag1_HA_color = lag1_period_data['color']
                lag2_HA_color = lag2_period_data['color']
                lag3_HA_color = lag3_period_data['color']
                lag5_HA_color = lag5_period_data['color']
                lag7_HA_color = lag7_period_data['color']
                # High/Low Bar Data
                cur_bar_data_high = cur_period_data['HA_high'] # Get the last row high price on the bar
                cur_bar_data_low = cur_period_data['HA_low'] # Get the last row high price on the bar
                lag1_bar_data_high = lag1_period_data['HA_high']
                lag1_bar_data_low = lag1_period_data['HA_low']
                lag2_bar_data_high = lag2_period_data['HA_high']
                lag2_bar_data_low = lag2_period_data['HA_low']
                # Rolling Avg High/Low Bar Data
                cur_rolling_avg_high = cur_period_data['rolling_avg_high_price'] #pulling from rolling_avg_low func
                cur_rolling_avg_low = cur_period_data['rolling_avg_low_price'] #pulling from rolling_avg_high func
                lag1_rolling_avg_high = lag1_period_data['rolling_avg_high_price']
                lag1_rolling_avg_low = lag1_period_data['rolling_avg_low_price']
                # Volume Bar Data
                cur_bar_volume = cur_period_data['volume']
                lag1_bar_volume = lag1_period_data['volume']
                lag2_bar_volume = lag2_period_data['volume']
                cur_rolling_avg_volume = cur_period_data['rolling_avg_volume']
                # RSI Bar Data
                cur_rsi = cur_period_data['RSI']
                lag1_rsi = lag1_period_data['RSI']
                lag2_rsi = lag2_period_data['RSI']
                # EMA/SMA Data
                cur_ema_fast = cur_period_data['EMA_Fast']
                cur_ema_slow = cur_period_data['EMA_Slow']
                cur_sma_fast = cur_period_data['SMA_Fast']
                cur_sma_slow = cur_period_data['SMA_Slow']
                lag1_ema_fast = lag1_period_data['EMA_Fast']
                lag1_ema_slow = lag1_period_data['EMA_Slow']
                lag1_sma_fast = lag1_period_data['SMA_Fast']
                lag2_sma_fast = lag2_period_data['SMA_Fast']
                lag2_ema_fast = lag2_period_data['EMA_Fast']
                cur_ema_mod = cur_period_data['EMA_Mod']
                cur_ema_20 = cur_period_data['EMA_20']
                # MACD Data
                cur_macd = cur_period_data['macd']
                lag1_macd = lag1_period_data['macd']
                lag2_macd = lag2_period_data['macd']
                cur_macd_signal = cur_period_data['signal']
                lag1_macd_signal = lag1_period_data['signal']
                lag2_macd_signal = lag2_period_data['signal']
                cur_macd_histogram = cur_period_data['histogram']
                lag1_macd_histogram = lag1_period_data['histogram']
                # Parabolic SAR Data
                cur_psar = cur_period_data['parabolic_sar']
                lag1_psar = lag1_period_data['parabolic_sar']
                lag2_psar = lag2_period_data['parabolic_sar']
                lag3_psar = lag3_period_data['parabolic_sar']
                # BB Data Intense
                cur_bb_upper_band_intense = cur_period_data['upper_band']
                lag1_bb_upper_band_intense = lag1_period_data['upper_band']
                lag2_bb_upper_band_intense = lag2_period_data['upper_band']
                cur_bb_lower_band_intense = cur_period_data['lower_band']
                lag1_bb_lower_band_intense = lag1_period_data['lower_band']
                lag2_bb_lower_band_intense = lag2_period_data['lower_band']
                # BB Data Tame
                cur_bb_upper_band_tame = cur_period_data['upper_b']
                lag1_bb_upper_band_tame = lag1_period_data['upper_b']
                cur_bb_lower_band_tame = cur_period_data['lower_b']
                lag1_bb_lower_band_tame = lag1_period_data['lower_b']
                # Stoch Data
                cur_stoch_SlowK = cur_period_data['SlowK']
                cur_stoch_SlowD = cur_period_data['SlowD']
                lag1_stoch_SlowK = lag1_period_data['SlowK']
                lag1_stoch_SlowD = lag1_period_data['SlowD']
                lag2_stoch_SlowK = lag2_period_data['SlowK']
                lag2_stoch_SlowD = lag2_period_data['SlowD']
                # VWAP
                cur_vwap = cur_period_data['vwap']
                # Avg HLC Data
                cur_avg_hlc = ((cur_bar_data + cur_bar_data_high + cur_bar_data_low)/3)
                lag1_avg_hlc = ((lag1_bar_data + lag1_bar_data_high + lag1_bar_data_low)/3)
                cur_avg_lc = ((cur_bar_data + cur_bar_data_low)/2)
                cur_avg_hc = ((cur_bar_data + cur_bar_data_high)/2)
                # Data Variables Assignment Completed
                
                print("\n")
                print(f"Latest data for {symbol}:")
                print("-----------------------------------------")
                print(f"Cur Bar Closing Price: {cur_bar_data.round(2)}")
                print(f"Lag 1 Bar Closing Price: {lag1_bar_data.round(2)}")
                print(f"Lag 2 Bar Closing Price: {lag2_bar_data.round(2)}")
                
                filled_qty_long = 0
                existing_long_pos = False
                filled_qty_short = 0
                existing_short_pos = False
                
                # The script verifies if there are any existing long or short positions to prevent holding simultaneous conflicting positions.
                if len(positions) > 0:
                    for position in positions:
                        if position.symbol == symbol and int(position.qty) > 0:
                            print("Existing long position of {} stocks in {}... skipping".format(position.qty, symbol))
                            existing_long_pos = True
                        elif position.symbol == symbol and int(position.qty) < 0:
                            print("Existing Short position of {} stocks in {}... skipping".format(position.qty, symbol))
                            existing_short_pos = True
                            
                # Entry Criteria Long:
                if not existing_long_pos and not existing_short_pos:
                    # LONG TRADE ENTRY LOGIC PLACED HERE
                    if ((lag1_macd < lag1_macd_signal and cur_macd > cur_macd_signal) and \
                        (lag1_stoch_SlowK < lag1_stoch_SlowD and cur_stoch_SlowK > cur_stoch_SlowD) and \
                        (lag1_HA_color == 'red' and cur_HA_color == 'green') and \
                        (cur_rsi < 50)):
                    # LONG TRADE ENTRY LOGIC ENDS HERE
                            try:
                                print(f"(ENTERING) (Buying) long signal generated for {symbol} at price {cur_bar_data.round(2)}")
                                qty = max(1, int(position_size / cur_bar_data))
                                order = api.submit_order(symbol,
                                                         qty,
                                                         side='buy',
                                                         type='market',
                                                         time_in_force='gtc')
                                print(f"(BOUGHT) Submitted limit buy order for {qty} shares in {symbol}")
                                combined_trade_count += 1
                            except Exception as e:
                                print(f"(FAILED) to Submit LONG signal for {symbol}. Error: {e}")
                                continue
                    else:
                        print(f"1) No Long Signals Generated for {symbol}...")
                
                # Entry Criteria Short:
                if not existing_long_pos and not existing_short_pos:
                    # SHORT TRADE ENTRY LOGIC PLACED HERE
                    if ((lag1_macd > lag1_macd_signal and cur_macd < cur_macd_signal) and \
                        (lag1_stoch_SlowK > lag1_stoch_SlowD and cur_stoch_SlowK < cur_stoch_SlowD) and \
                        (lag1_HA_color == 'green' and cur_HA_color == 'red') and \
                        (cur_rsi > 50)):
                    # SHORT TRADE ENTRY LOGIC PLACED HERE
                            try:
                                print(f"(ENTERING) (Selling) short signal generated for {symbol} at price {cur_bar_data.round(2)}")
                                qty = max(1, int(position_size / cur_bar_data))
                                order = api.submit_order(symbol, 
                                                         qty, 
                                                         side='sell',
                                                         type='market',
                                                         time_in_force='gtc')
                                print(f"(BOUGHT) sell order for {qty} shares in {symbol}")
                                combined_trade_count += 1
                            except Exception as e:
                                print(f"(FAILED) to Submit SHORT signal for {symbol}. Error: {e}")
                                continue  # Go to next iteration if trade cannot be placed (i.e. cannot short error)
                    else:
                        print(f"2) No Short Signals Generated for {symbol}...")
                        
                # Logs information for each iteration, such as current positions and the number of trades executed.
                print(f"Total trades for this script iteration: {combined_trade_count}")
                print("-----------------------------------------")

            # Calculate seconds until the next minute starts
            # if using 2m,5m, etc. bars, minutes will need to match the timeframe value below
            current_time = datetime.now()
            next_min = (current_time + timedelta(minutes=1)).replace(second=0, microsecond=0)
            
            # Puts the script to sleep for a calculated duration until the next minute, ensuring the script operates efficiently.
            # Calculate sleep time (1 - 30 seconds) till after the start of the next minute. See polling_interval in global variable settings
            sleep_time = (next_min - current_time).seconds + polling_interval
            
            if combined_trade_count >= max_trade_attempts:
                time.sleep(100)
                print('MAX TRADE ATTEMPTS REACHED, PLEASE TURN OFF THE CODE UNTIL THE NEXT TRADING DAY.')
                print('KEEP TESTING!!! :D')
            print(f"Max trade attempts for this script iteration: {max_trade_attempts}")
            print(f"The number of tickers in this list are {num_of_symbols}")
            print("\n")
            print(f"This is a {bot_type} bot...")
            print(f"This is the {strat_name} Strategy.")
            portfolio = api.list_positions()
            print("\n")
            print("CURRENT OPEN POSITIONS:")
            for position in portfolio:
                print("{} shares of {}".format(position.qty, position.symbol))
            print("\n")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("\n")
            print(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            
        else:
            print("Outside of trading hours. Current time: {}. Sleeping for 60 seconds.".format(datezone_est.strftime("%Y-%m-%d %H:%M:%S")))
            time.sleep(60)
        
if __name__ == '__main__':
    main()

#################Code Complete#################

 




