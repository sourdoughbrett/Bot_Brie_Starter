## Introduction
If you have reached this page that is an excellent sign and you are now ready to put your bot to the real test :D

Firstly, I want to say thank you for purchasing this product. I hope you find immense value in this code and can use it for whatever purposes you deem fit.
This code is designed to be refactored, modularized, altered and used in ANY way you see fit. This is a boilerplate product, an outline or shell if you will. As comprehensive as it is, it is still may be missing a feature or two you deem critical and important. 

There are no limitations to what you can do with the boilerplate, I encourage you to make the bot better if you see a path to do so.
With that said, Chat GPT and other AI tools can be your best friend in 1) Adding features to the boilerplate 2) Troubleshooting code errors 3) Explaining what each aspect of the code is doing
Leverage Chat GPT to answer nuanced questions about your Bot.

If you are new to algorithmic trading, this is an excellent start and will push you immeasurably forward in your algorithmic trading journey. 

## 1. Getting Started
The code is setup to plug and play (update your strategies of course). To begin, you need to set up your API keys:
1.	Open the `boilerplate_config.yml` file located in the `/config` directory.
2.	Add Alpaca Keys to yml file
```plaintext
# Alpaca Keys
API_KEY: ''
SECRET_KEY: ''
api_base_url: 'https://paper-api.alpaca.markets'
```
3.	Update ‘config_file_path’ in `scripts/boilerplate/` to yml filepath (ex: "C:/Users/.../..../..../boilerplate_config.yml") 
4.	Test the bot by running the program (Press F5).
     - If any errors are produced it likely has to do with the timeframe, start_date and end_date. Make sure the boundaries are appropriate depending on the timeframe used.
  
As laid out in the README file, the boilerplate is broken down into 5 sections:
```plaintext
Section 1: Config, Modules, and Imports
Section 2: Global/Key Variable Settings
Section 3: Key Functions
Section 4: Test DataFrame Config
Section 5: Main Function
```
The only sections you will need to modify (unless your adding indicators or testing how the df will update is section 2 and section 5).
  
## 2. Adusting Global Parameters for Script Optimization
Section 2 is where you will update your start/end times, indicator values, and trailing stop params.

```plaintext
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

## 3. Configuring and running sample strategies

## 4. Adding more Bars and Indicators

## 5. Backtesting Overview
