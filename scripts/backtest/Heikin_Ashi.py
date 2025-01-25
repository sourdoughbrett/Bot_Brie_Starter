# Heikin Ashi Back Tester

strat_name = 'back-tester_heikin-ashi'
config_file_path = ""
# "C:/Users/.../..../..../backtest_config.yml"

#####################################Section 1: Config, Modules, and Imports############################

# Modules

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import talib
import yaml
from tqdm import tqdm


# Load YAML configuration
with open(config_file_path, "r") as file:
    config = yaml.safe_load(file)

# Access variables
API_KEY = config['API_KEY']
SECRET_KEY = config["SECRET_KEY"]
api_base_url = config["api_base_url"]
symbols = config["symbols"] #remove this if adding locally, add symbols = [''] below.
start_time = config["start_time"]
end_time = config["end_time"]
tp_pct = config["tp_pct"] # Set your take profit of your backtest
sl_pct = config["sl_pct"]  # Set your stop loss of your backtest
initial_balance = config["ib"] # initial balance

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


###################################Section 2: Global/Key Variable Settings#####################################

# add symbols here if you do not want to store them via YML config file
# symbols = ['NVDA','TSLA','MSFT']

'''
TIME VALUES (UPDATE BASED ON STRATEGY)
'''

# Timeframe constraints ("Minute","Hour","Day") ... See 5Min bar example file or alpaca docs if looking to implement 5m or custom bar timeframes.

timeframe = "Day"


'''
INDICATOR VALUES (UPDATE BASED ON STRATEGY)

Only a few indicator values have been hard coded as global variables, you can add as many as you would like :)
'''

# MACD (12,26,4-5) is optimal
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

# Retrieving Heiken Ashi Bar Data (Essentially the "average" bar price, with an associated bar color, red or green)
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

# TA-Lib Functions
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

        
###################################Section 4: Test DataFrame Config####################################
# Feature Engineering
# Concat all indicator values into one DF = hist_data()
#----------------------------------------------#
hist_data_raw = get_stock_bars_data(api_key=API_KEY,secret_key=SECRET_KEY,tickers=symbols,start_date=start_time,end_date=end_time)
heiken_ashi_df = calculate_heiken_ashi(df=hist_data_raw, open_col='open', high_col='high', low_col='low', close_col='close')
hist_data_raw['HA_open'] = heiken_ashi_df['HA_open']
hist_data_raw['HA_high'] = heiken_ashi_df['HA_high']
hist_data_raw['HA_low'] = heiken_ashi_df['HA_low']
hist_data_raw['HA_close'] = heiken_ashi_df['HA_close']
hist_data_raw['color'] = heiken_ashi_df['color']
#MACD concat
macd_df = calculate_macd(df=hist_data_raw,column='HA_close',fast_period=macd_fast_period_val, slow_period=macd_slow_period_val, signal_period=macd_signal_period_val)
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
sar_df = calculate_parabolic_SAR(df=hist_data_raw, column='HA_close',acceleration=psar_acceleration_dev_factor, maximum=psar_maximum_dev_factor)
hist_data_raw['parabolic_sar'] = sar_df['parabolic_sar']
#Stochastic concat
stochastic_df = calculate_stochastic(df=hist_data_raw, high_col='HA_high',low_col='HA_low',close_col='HA_close',fastk_period=stoch_fastK_period_val,slowk_period=stoch_slowK_period_val, slowd_period=stoch_slowD_period_val)
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
sma_series = calculate_sma_fast(df=hist_data_raw, column='HA_close',period=sma_fast_period_val)
hist_data_raw['SMA_Fast'] = sma_series
#EMA_Fast Concat
ema_series_fast = calculate_ema_fast(df=hist_data_raw, column='HA_close',period=ema_fast_period_val)
hist_data_raw['EMA_Fast'] = ema_series_fast
#EMA_Slow Concat
ema_series_slow = calculate_ema_slow(df=hist_data_raw, column='HA_close',period=ema_slow_period_val)
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


test_data = hist_data_raw.reset_index()
print(test_data)

# Ensure the timestamp column is a datetime type
test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
test_data.set_index('timestamp', inplace=True)

#test_data['close'] = test_data['close'].astype(float).values
print("\nStarting backtest...")  # Debugging print

###################################Section 5: Backtesting#####################################

'''

This next block of code is where we will manually add our code logic.

make sure to add bars. The rest of the data we can access from the hist_data_raw df which becomes our test_data df in the backtest.

position size is set for half of the initial balance (ib)

'''

# Define the trading logic
def generate_signals(df):
    """Generate buy and sell signals based on the provided trading logic."""
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0

    for i in range(0, len(df)):
        lag2 = df.iloc[i - 3] # lag 2 bar
        lag1 = df.iloc[i - 2] # lag 1 bar
        cur = df.iloc[i - 1] # current bar

        # LONG BACKTEST LOGIC PLACED HERE
        if ((lag1['macd'] < lag1['signal'] and cur['macd'] > cur['signal']) and
            (lag1['color'] == 'red' and cur['color'] == 'green') and
            (cur['SlowD'] > 80 and cur['SlowK'] > 80) and
            (cur['SlowK'] > cur['SlowD'])):
            # LONG BACKTEST LOGIC ENDS HERE
            signals.at[cur.name, 'signal'] = 1
            
        # SHORT BACKTEST LOGIC PLACED HERE
        if ((lag1['macd'] > lag1['signal'] and cur['macd'] < cur['signal']) and
            (lag1['color'] == 'green' and cur['color'] == 'red') and
            (cur['SlowD'] < 80 and cur['SlowK'] < 80) and
            (cur['SlowD'] > 20 and cur['SlowK'] > 20) and
            (cur['SlowK'] < cur['SlowD'])):
            # SHORT BACKTEST LOGIC ENDS HERE
            signals.at[cur.name, 'signal'] = -1 

    return signals

###################################Section 6: Processing#####################################

##########################################do not alter code below unless you need to :D################################################
def backtest_strategy(df, signals, initial_balance=initial_balance, take_profit=tp_pct, stop_loss=sl_pct):
    """Backtest the trading strategy and calculate performance metrics."""
    position_size = initial_balance * .5 # Position Size is set for half the initial balance
    positions = 0
    trades = []
    wins = 0
    losses = 0
    cumulative_pnl = []  # Track cumulative PnL over time
    pnl = 0  # Total profit/loss
    
    for i in tqdm(range(len(signals))):
        signal = signals.iloc[i]['signal']
        cur_close = df.iloc[i]['close']

        # Long position entry
        if signal == 1 and positions == 0:
            entry_price = cur_close
            num_shares = position_size // cur_close
            positions += num_shares
            trades.append((df.index[i], 'BUY', cur_close, num_shares))
        
        # Short position entry
        elif signal == -1 and positions == 0:
            entry_price = cur_close
            num_shares = position_size // cur_close
            positions -= num_shares
            trades.append((df.index[i], 'SELL', cur_close, num_shares))

        # Exit for long position
        if positions > 0:
            pnl_percentage = (cur_close - entry_price) / entry_price
            if pnl_percentage >= take_profit or pnl_percentage <= -stop_loss or signal == -1:
                pnl += positions * (cur_close - entry_price)
                if pnl_percentage >= take_profit:
                    wins += 1
                else:
                    losses += 1
                positions = 0
                trades.append((df.index[i], 'SELL', cur_close, num_shares))
        
        # Exit for short position
        elif positions < 0:
            pnl_percentage = (entry_price - cur_close) / entry_price
            if pnl_percentage >= take_profit or pnl_percentage <= -stop_loss or signal == 1:
                pnl += abs(positions) * (entry_price - cur_close)
                if pnl_percentage >= take_profit:
                    wins += 1
                else:
                    losses += 1
                positions = 0
                trades.append((df.index[i], 'BUY', cur_close, abs(num_shares)))
        
        # Track cumulative PnL
        cumulative_pnl.append(pnl)
    
    # Calculate performance metrics
    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    strategy_return_pct = (((initial_balance + pnl) / initial_balance) - 1) * 100  # Convert to percentage
    
    # Summary
    print("\n--- Backtest Summary ---")
    print(f"for {symbols}")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Position Size: ${position_size:.2f}")
    print(f"Take profit %: {tp_pct}")
    print(f"Stop loss %: {sl_pct}")
    print("-- Performance Summary --")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {wins}")
    print(f"Losing Trades: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit: ${pnl:.2f}")
    print(f"Cumulative Return Pct: {strategy_return_pct:.2f}%")

    return trades, win_rate, pnl, df.index[:len(cumulative_pnl)], cumulative_pnl


def visualize_strategy_performance(dates, cumulative_pnl):
    """Visualize the strategy performance as a time graph."""
    # Convert dates to datetime if not already
    if isinstance(dates[0], tuple):  # Handle MultiIndex
        dates = [d[1] for d in dates]
    plt.figure(figsize=(14, 7))
    plt.plot(dates, cumulative_pnl, label="Cumulative PnL", color='blue', linewidth=2)
    plt.axhline(0, color='red', linestyle='--', label="Break-Even Line")
    plt.title(f"Strategy Performance Over Backtest Period for {symbols}", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative PnL ($)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Generate trading signals
signals = generate_signals(hist_data_raw)

# Backtest the strategy
trades, win_rate, total_strategy_profit, dates, cumulative_pnl = backtest_strategy(hist_data_raw, signals)

# Visualize strategy performance
visualize_strategy_performance(dates, cumulative_pnl)
