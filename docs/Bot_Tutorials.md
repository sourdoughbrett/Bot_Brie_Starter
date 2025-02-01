## Introduction 👏
If you have reached this page that is an excellent sign and you are now ready to put your bot to the real test :D

Firstly, I want to say thank you for purchasing this product. I hope you find immense value in this code and can use it for whatever purposes you deem fit.
This code is designed to be refactored, modularized, altered and used in ANY way you see fit. This is a boilerplate product, an outline or shell if you will. As comprehensive as it is, it is still may be missing a feature or two you deem critical and important. 

There are no limitations to what you can do with the boilerplate, I encourage you to make the bot better if you see a path to do so.
With that said, Chat GPT/DeepSeek and other AI tools can be your best friend in 1) Adding features to the boilerplate 2) Troubleshooting code errors 3) Explaining what each aspect of the code is doing
Leverage Chat GPT to answer nuanced questions about your Bot.

If you are new to algorithmic trading, this is an excellent start and will push you immeasurably forward in your algorithmic trading journey.

---

## 1. Getting Started 🚀
The code is setup to plug and play (update your strategies of course). To begin, you need to set up your API keys:
1.	Open the `boilerplate_config.yml` file located in the `/config` directory.
2.	Add Alpaca Keys to yml file
```plaintext
API_KEY: ''
SECRET_KEY: ''
api_base_url: 'https://paper-api.alpaca.markets'
```
3.	Update ‘config_file_path’ in `scripts/boilerplate/` to yml filepath (ex: "C:/Users/.../..../..../boilerplate_config.yml") 
4.	Test the bot by running the program (Press F5) within the Spyder IDE. Alternatively you can run the script via the Anaconda Prompt, Command Line, or your own IDE system.
     - If any errors are produced it likely has to do with the timeframe, start_date and end_date. Make sure the boundaries are appropriate depending on the timeframe used.
  
As laid out in the README file, the boilerplate is broken down into 5 sections:
```plaintext
Section 1: Config, Modules, and Imports
Section 2: Global/Key Variable Settings (Add settings here)
Section 3: Key Functions
Section 4: Test DataFrame Config
Section 5: Main Function (Add strategies here)
```
The only sections you will need to modify (unless your adding indicators or testing how the test df (hist_data_raw) will update is section 2 and section 5).

---
  
## 2. Adusting Global Parameters for Script Optimization 🧑‍💻
Open the `/Trailing_Stop_OHLC.py` file located in the `/scripts/boilerplate` directory.

Section 2 is where you will update your start/end times, indicator values, and trailing stop params.

```plaintext
###################################Section 2: Global/Key Variable Settings#####################################

# add symbols here if you do not want to store them via YML config file
# symbols = ['NVDA','TSLA','MSFT']

'''
TIME VALUES (UPDATE BASED ON STRATEGY)

must make sure yml date file is updated as its pulling the data from X amt of days back from your indicator vals.
'''

timeframe = "Hour" #Change to Minute, Hour, Day, Week, Month for 1m, 1h, 1d, 1w, 1m timeframes.

# Start/End Time constraints (Start Time 9,30 = market open9:30am est)
hour_start_time = 9
minute_start_time = 30
hour_end_time = 15
minute_end_time = 30

# elapsed bar time (how many seconds until after the bar data updates do you want the script to update?) Recommendation is 1 - 3 seconds.
elapsed_bar_time = 1

'''
INDICATOR VALUES (UPDATE BASED ON STRATEGY)

Only a few indicator values have been hard coded as global variables, you can add as many as you would like :)
'''

# MACD 
macd_fast_period_val = 12
macd_slow_period_val = 26
macd_signal_period_val = 2
# EMA
ema_fast_period_val = 3
ema_mod_period_val =  6
ema_slow_period_val = 50
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
trail_pct = 2.0
```

---

## 3. Configuring and running your Trading Strategy 📊
Section 5 is where you will provide your strategy logic.

This strategy takes a position when the PSAR (trend following indicator) changes direction while the RSI is underneath or over a certain threshold.

```plaintext
# LONG TRADE ENTRY LOGIC PLACED HERE
if ((cur_bar_data > cur_psar and lag1_bar_data < lag1_psar) and \
    (cur_rsi < 30)):
# LONG TRADE ENTRY LOGIC ENDS HERE

code...

# SHORT TRADE ENTRY LOGIC PLACED HERE
if ((cur_bar_data < cur_psar and lag1_bar_data > lag1_psar) and \
    (cur_rsi > 70)):
# SHORT TRADE EXIT LOGIC PLACED HERE
```
So what exactly is going on?

When the script runs it updates every 60 seconds + the value of the "elapsed_bar_time" variable. In this case, 1 second after the minute bar elapses or updates. The main_function has a while loop condition that runs and checks the api for updates on this cycle pattern. 

If you wanted to update the script more or less frequently, you would change next_min to your desired timeframe (ex: every 15 seconds, every 15 minutes, etc.).
```plaintext
# Calculate seconds until the next minute starts
# if using 2m,5m, etc. bars, minutes will need to match the timeframe value below
current_time = datetime.now()
next_min = (current_time + timedelta(minutes=1)).replace(second=0, microsecond=0)
```
⏰
Moving forward..!

Let's say I wanted to check if the RSI was underneath 10 or over 90 for 3 consecutive periods (wow oversold much?)
```plaintext
# LONG TRADE ENTRY LOGIC PLACED HERE
if ((lag2_rsi < 10 and lag1_rsi < 10 and cur_rsi < 10)):
# LONG TRADE ENTRY LOGIC ENDS HERE

code...

# SHORT TRADE ENTRY LOGIC PLACED HERE
if ((lag2_rsi > 90 and lag1_rsi > 90 and cur_rsi > 90)):
# SHORT TRADE EXIT LOGIC PLACED HERE
```
This can become quite useful, especially if you scan for dozens of assets at a time to determine if any anomalistic conditions are occuring in real-time.

Finding confluences of signals can be lead to better outcomes as well:
```plaintext
# LONG TRADE ENTRY LOGIC PLACED HERE
if ((lag1_macd < lag1_macd_signal and cur_macd > cur_macd_signal) and \
    (lag1_stoch_SlowK < lag1_stoch_SlowD and cur_stoch_SlowK > cur_stoch_SlowD) and \
    (lag1_HA_color == 'red' and cur_HA_color == 'green')):
# LONG TRADE ENTRY LOGIC ENDS HERE

code...

# SHORT TRADE ENTRY LOGIC PLACED HERE
if ((lag1_macd > lag1_macd_signal and cur_macd < cur_macd_signal) and \
    (lag1_stoch_SlowK > lag1_stoch_SlowD and cur_stoch_SlowK < cur_stoch_SlowD) and \
    (lag1_HA_color == 'green' and cur_HA_color == 'red')):
# SHORT TRADE ENTRY LOGIC PLACED HERE
```

Have an idea? Build it, Test it, Validate it.

---

## 4. Adding more Bars and Indicators 🔎

Section 3 is where you will add more indicators if you desire. If you have an idea, build your own function, ask chat gpt to help build a function, etc.

Example:
1) Copy the MACD function
2) Begin to write in Chat GPT/Deepseek.. "Please create "xyz" indicator (LazyBear Momentum) from "zyx" (TradingView/TA-Lib) and format function precisely like my MACD function, here is my MACD function: (paste MACD function)"

Chat GPT/Deepseek Output:
```plaintext
def calculate_momentum(df, column="close", bollinger_period=20, keltner_period=20, momentum_period=14):
    """
    Calculate the LazyBear Momentum Indicator using Bollinger Bands, Keltner Channels, and Momentum Oscillator.

    Args:
        df (pandas.DataFrame): DataFrame containing the price data.
        column (str): Name of the column in the DataFrame that contains the prices. Default is "HA_close".
        bollinger_period (int): Number of periods to consider for Bollinger Bands. Default is 20.
        keltner_period (int): Number of periods to consider for Keltner Channels. Default is 20.
        momentum_period (int): Number of periods to consider for the Momentum Oscillator. Default is 14.

    Returns:
        pandas.DataFrame: DataFrame containing the Squeeze Indicator and Momentum Histogram.
    """
    # Calculate Bollinger Bands
    df["bollinger_mid"] = df[column].rolling(window=bollinger_period).mean()
    df["bollinger_std"] = df[column].rolling(window=bollinger_period).std()
    df["bollinger_upper"] = df["bollinger_mid"] + 2 * df["bollinger_std"]
    df["bollinger_lower"] = df["bollinger_mid"] - 2 * df["bollinger_std"]

    # Calculate Keltner Channels
    df["keltner_mid"] = df[column].rolling(window=keltner_period).mean()
    df["keltner_atr"] = df["bollinger_std"].rolling(window=keltner_period).mean()  # Approximation for ATR
    df["keltner_upper"] = df["keltner_mid"] + 1.5 * df["keltner_atr"]
    df["keltner_lower"] = df["keltner_mid"] - 1.5 * df["keltner_atr"]

    # Squeeze Indicator: Bollinger Bands inside Keltner Channels
    df["squeeze_on"] = (
        (df["bollinger_upper"] < df["keltner_upper"]) & 
        (df["bollinger_lower"] > df["keltner_lower"])
    )

    # Momentum Oscillator
    df["momentum"] = df[column].diff(momentum_period)

    # Signal: Color Histogram for Momentum
    df["momentum_histogram"] = df["momentum"].rolling(window=5).mean()

    # Select and return relevant columns
    momentum_df = pd.DataFrame({
        "bollinger_upper": df["bollinger_upper"],
        "bollinger_lower": df["bollinger_lower"],
        "keltner_upper": df["keltner_upper"],
        "keltner_lower": df["keltner_lower"],
        "squeeze_on": df["squeeze_on"],
        "momentum_histogram": df["momentum_histogram"]
    }, index=df.index)

    return momentum_df
```

4) Verify Output is valid and the function returns a DataFrame or Series (momentum_df).
5) Append momentum_df to the hist_data_raw df for testing.

```plaintext
momentum_df = calculate_momentum(df=hist_data_raw, column="close", bollinger_period=20, keltner_period=20, momentum_period=14)
hist_data_raw['bollinger_upper'] = momentum_df['bollinger_upper']
hist_data_raw['bollinger_lower'] = momentum_df['bollinger_lower']
hist_data_raw['keltner_upper'] = momentum_df['keltner_upper']
hist_data_raw['keltner_lower'] = momentum_df['keltner_lower']
hist_data_raw["squeeze_on"] = momentum_df["squeeze_on"]
hist_data_raw['momentum_histogram'] = momentum_df['momentum_histogram']
```

6) Run script, check the hist_data_raw df to ensure all data is updated properly.
7) If hist_data_raw df is valid, then append to stock_data within the main function of section 5 and begin to create your strategies around this indicator (E.G. if squeeze_on == True, go long).
8) Test in main_func before deploying. Always test thoroughly...

---

Bars is pretty straight forward. All data is appended to symbol_data (this includes all bar and indicator information, everything).

1) Add your bars:
```plaintext
 symbol_data = stock_data.loc[symbol] # Get data for the current symbol (symbol data is where ALL of our data will be stored)
 cur_period_data = symbol_data.iloc[-1] # Getting most recent row of data (Creating the timeseries data by time increment relative to real-time, these will become our bar increments)
 lag1_period_data = symbol_data.iloc[-2] # Delayed by 1 periods
 lag2_period_data = symbol_data.iloc[-3] # Delayed by 2 periods

 cur_bar_data = cur_period_data['HA_close'] # Get the last row (latest period)
 lag1_bar_data = lag1_period_data['HA_close']

 cur_HA_color = cur_period_data['color']
 lag1_HA_color = lag1_period_data['color']

 cur_bar_data_high = cur_period_data['HA_high'] # Get the last row high price on the bar
 cur_bar_data_low = cur_period_data['HA_low'] # Get the last row high price on the bar
```

2) Make your own bars:
```plaintext
 # Avg HLC Data
 cur_avg_hlc = ((cur_bar_data + cur_bar_data_high + cur_bar_data_low)/3)
 lag1_avg_hlc = ((lag1_bar_data + lag1_bar_data_high + lag1_bar_data_low)/3)
 cur_avg_lc = ((cur_bar_data + cur_bar_data_low)/2)
 cur_avg_hc = ((cur_bar_data + cur_bar_data_high)/2)
```

3) Get funky:
```plaintext
 # Rolling Avg High/Low Bar Data
 cur_rolling_avg_high = cur_period_data['rolling_avg_high_price'] #pulling from rolling_avg_low func
 cur_rolling_avg_low = cur_period_data['rolling_avg_low_price'] #pulling from rolling_avg_high func
 lag1_rolling_avg_high = lag1_period_data['rolling_avg_high_price']
 lag1_rolling_avg_low = lag1_period_data['rolling_avg_low_price']
```

---

## 5. Backtesting Overview 💻

1.	Open the `backtest_config.yml` file located in the `/config` directory.
2.	Add Alpaca Keys to yml file, update ib (initial balance), tp_pct, sl_pct, and date times.
        - The backtesting chart is designed for 1 symbol per backtest, encouraged to test 1 symbol at a time.
3.   Open the `OHLC.py` file located in the `/scripts/backtest/` directory.
4.   Update ‘config_file_path’ in `/scripts/backtest/` to yml filepath (ex: "C:/Users/.../..../scripts/backtest_config.yml").

The structure of the back-testing script is the exact same as the boilerplate script except section 5 now becomes where we add our strategy logic.

Within the generate_signals function, you will add your strategy logic. Here is an example of creating a backtest with the follow crossover strategy with the MACD and Heikin Ashi.

```plaintext
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
            (lag1['color'] == 'red' and cur['color'] == 'green')):
            # LONG BACKTEST LOGIC ENDS HERE
            signals.at[cur.name, 'signal'] = 1
            
        # SHORT BACKTEST LOGIC PLACED HERE
        if ((lag1['macd'] > lag1['signal'] and cur['macd'] < cur['signal']) and
            (lag1['color'] == 'green' and cur['color'] == 'red')):
            # SHORT BACKTEST LOGIC ENDS HERE
            signals.at[cur.name, 'signal'] = -1 

    return signals
```
Now, I can run the backtest on different timeframes, tp/sl parameters, indicator values, and datetimes to test for efficacy.

To add more bars for backtesting, add lag3, lag4, lag5, etc. with the same formatting as cur, lag1, ang lag2 above.

Align with your strategy idea to hone in on a profitable idea or system 🎯

## Bonus (Build your ML Algorithm) 🚧

By accessing the Alpaca API users have access to thousands of rows and columns of historical bar data. Perfect for building an LSTM model for timeseries predictions.

What if you could build a model that updated every hour that was forecasting the price of a particular asset for the next hour? What if efficacy can be found based on more data points? How would this transform your strategies?

To be announced!
