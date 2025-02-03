# **Bot Brie Starter Boilerplate**

clone the repository to your local environment:
```plaintext
git clone https://github.com/sourdoughbrett/Bot_Brie_Starter.git
```

Alternatively you can copy and paste the files from Github since there are only a hand full.

--

To begin using the Bot Brie All-In Boilerplate, you'll first need access to Alpaca’s API. 

Follow these steps to create an Alpaca account and obtain your API keys:

## **Pre-Requisite: Create an Alpaca Account** 🌐
1. Visit [Alpaca's Website](https://alpaca.markets/) and create an account.  
   - If you're new to Alpaca, consider starting with a **paper trading account** to test strategies risk-free.
   - Paper trading accounts provide virtual money, allowing you to familiarize yourself with the platform.

2. Generate your **API Key** and **Secret Key** from the Alpaca dashboard:  
   - Log in to your Alpaca account.  
   - Navigate to the **API Keys** section.  
   - Copy the keys and store them securely. These keys will be required to connect to the API.  

3. Familiarize yourself with Alpaca's documentation for API usage:  
   - [Alpaca API Docs](https://alpaca.markets/docs/)

---

## **1️⃣ Overview** 📣
The Bot Brie All-In Boilerplate is a powerful, customizable trading framework designed to help you automate trading strategies using Alpaca’s API. This package supports:
- **Trailing_Stops** for dynamic risk management.
- **Multiple Timeframes** for flexible strategy implementation (e.g., 1m, 15m, 1h, daily bars).
- **Heikin Ashi Bars** for flexible strategy implementation (e.g., 1m, 15m, 1h, daily bars).
- **Comprehensive Backtesting Tools** for strategy validation before live trading.

Explore scripts for both live/paper trading and backtesting in the respective `scripts/boilerplate` and `scripts/backtest` directories.

Remember, always thoroughly forward and backtest for strategy efficacy before launching live. 

*Test ==> Feedback ==> Iterate ==> Improve*

---

## **2️⃣ Environment Setup** 🧑‍💻
To get started, ensure you have the required dependencies and tools installed. This includes setting up your Python environment with Anaconda and configuring the Spyder IDE for efficient coding and testing.

👉 **[View Full Environment Setup Guide](docs/Environment_Setup.md)**

---

## **3️⃣ Project Structure** 🚀
The boilerplate AND back-testing scripts timeframes can be customized (1m, 15m, 1h, daily, weekly, etc.). 

**Remember you need an Alpaca Market Subscription to access historical bars within 15 minutes of real-time data.**

The repository is organized as follows:

```plaintext
Starter-Package-Repo/
├── scripts/
│   ├── boilerplate/
│   │   ├── Trailing_Stop_OHLC.py        
│   │   ├── Heikin_Ashi_Swing_Alert.py  
│   ├── backtest/
│   │   ├── OHLC.py                
│   │   ├── Heikin_Ashi.py         
├── config/
│   ├── boilerplate_config.yml    
│   ├── backtest_config.yml     
├── docs/
│   ├── Environment_Setup.md     
│   ├── Bot_Tutorials.md       
├── README.md                
├── requirements.txt         
├── LICENSE         
└── CHANGELOG.md
```

### **A. Boilerplate Scripts**
Live trading scripts located in `scripts/boilerplate/`:
- **Trailing_Stop_OHLC.py**: Implements trailing stop logic for live trading (hourly timeframe example)
- **Heikin_Ashi_Swing_Alert.py**: Example script of a swing alert strategy with Heikin Ashi bars (daily timeframe example).

### **B. Backtesting Scripts**
Backtesting scripts located in `scripts/backtest/`:
- **OHLC.py**: Backtest strategies on traditional OHLC bars (hourly timeframe).
- **Heikin_Ashi.py**: Backtest strategies on Heikin Ashi bars (daily timeframe).

### **C. Configuration Files**

The `config/` directory contains YAML configuration files that store critical settings and parameters used by the scripts. These files help centralize and streamline the configuration process, making it easier to manage and update your trading environment. Key details include:

- **boilerplate_config.yml**:  
  - This file contains shared settings for live trading, such as your **API keys**, which are required to authenticate with Alpaca’s API.
  - It also stores **global variables** such as the list of trading symbols, position sizing, timeframe constraints, and indicator parameters.
  - By editing this file, you can customize the behavior of the live trading scripts without modifying the code directly.
- **backtest_config.yml**:  
  This file defines the parameters for running backtesting scenarios, such as historical start and end dates, the data source for historical bars, and any specific strategy configurations. It ensures consistency and reproducibility when testing strategies against historical data.

**Note:** To protect sensitive information like API keys, always keep these configuration files secure and avoid sharing them publicly. It’s recommended to use `.gitignore` to exclude these files from version control if you are using Git for your project.


### **D. Documentation Files**

The `docs/` directory contains detailed guides and reference materials to help users get started and maximize the potential of the Bot Brie Boilerplate:

- **`README.md`**: Provides an overview of the boilerplate, including a breakdown of project structure, environment setup, and support options.

- **`Environment_Setup.md`**: Step-by-step guide for setting up your environment, including:
  - Installing the Anaconda distribution.
  - Configuring the Spyder IDE.
  - Verifying and managing Python dependencies.

- **`Bot_Tutorials.md`**: Practical examples and tutorials for implementing various trading strategies, including:
  - Getting started with the Bot.
  - Configuring and running sample strategies.
  - Creating and customizing backtesting scenarios.

**How to Use the Documentation:**
- Start with `Environment_Setup.md` to prepare your development environment.
- Refer to `Bot_Tutorials.md` for hands-on guidance with live trading and backtesting strategies.
- Use `README.md` as a quick reference for navigating the repository and understanding its components.

**Note:** These guides are designed to ensure users, regardless of experience level, can effectively use the boilerplate to build and execute their trading strategies.

---

## **4️⃣ Boilerplate/Backtest Sections** 📚
All boilerplate AND back-testing scripts have the same structure. 

Below is a breakdown of the main intra-script code sections:

### **Section 1: Installing Modules and Imports**
- Import all packages and dependencies.
- Includes YML configuration for API keys and user-selected global variables.
- Set paper=True or False for paper/live trading (update api_base_url accordingly).

### **Section 2: Global/Key Variable Settings**
- Define key variables such as:
  - Timeframes (1m, 15m, 1h, 1d, 1w)
  - Start/End Times (down to the millisecond)
  - Indicator values
  - Trailing Stop Percentage
  - Elapsed Bar Time

### **Section 3: Key Functions**
- **Bar Retrieval**: Functions to retrieve historical bars (e.g., Heiken Ashi for All-In version).
- **Indicators**: Functions for MACD, RSI, Stochastic Oscillator, Bollinger Bands, and more.
- **Rolling Averages**: Utility functions for calculating rolling averages.
- **Supportive Functions**: Helper functions for trading logic and API integration.

### **Section 4: Test DataFrame Config (hist_data_raw)**
- Append all indicator and bar data to the `hist_data_raw` DataFrame.
- Access the DataFrame in Spyder to ensure data updates correctly.
- Prints the DataFrame output in the script.

### **Section 5: Main Function**
- ***For Live/Paper Trading:***
- **Market Hours:** The script continuously checks the current time against the defined market start and end times in the Eastern timezone. Trading operations only occur within these hours.
- **Data Retrieval:** Calls the Alpaca API to fetch historical stock data. Appends all data to `stock_data` for real-time processing.
- **Open Positions Check:** The script verifies if there are any existing long or short positions to prevent holding simultaneous conflicting positions.
- **Trade Execution Logic:** Implements trading logic (Long and Short) based on predefined conditions.
- **Order Status Check:** Continuously monitors the status of placed orders to ensure they are filled or canceled.
- **Sleep Timer:** Puts the script to sleep for a calculated duration until the next minute, ensuring the script operates efficiently.
- **Iteration Tracking:** Logs information for each iteration, such as current positions and the number of trades executed.
- Within the main function is where you will add the code logic for your strategies 🧑‍💻 visit `Bot_Tutorials.md`
- ***For Backtesting scripts:***
  - Contains backtesting functions to analyze historical strategy performance.
  - Creates a visual with matplotlib of the PnL performance over the backtest period.
  - Recommended to run a single asset in the backtest at a time.
  - Play with settings (tp,sl, position size, indicator vals, etc.).
  - *Backtesting logic should perfectly match paper/live strategy logic for highest accuracy of results and efficacy.*

---

## **5️⃣ Bot Tutorials** 💻
Learn how to leverage the boilerplate to design and test your own trading strategies. The Bot Tutorials guide provides step-by-step instructions for:
- Getting Started.
- Adjusting global parameters for script optimization.
- Configuring and running sample strategies.
- Adding bars and more indicators.
- Backtesting strategies for robust validation.

👉 **[View Full Strategy Tutorial Guide](docs/Bot_Tutorials.md)**

---

## **6️⃣ Support and Troubleshooting** 🛡️
If you encounter any issues or have questions, please refer to our troubleshooting guide or reach out to our support team.

### **Common Troubleshooting Tips:**
- **Configuration Errors**: Ensure all necessary values are correctly set in the YAML configuration files (`boilerplate_config.yml`, `backtest_config.yml`).
- **Dependency Issues**: Verify you have installed all required Python dependencies from the `requirements.txt` file.
- **Strategy Errors**: Thoroughly backtest and forward test strategies before deploying live.

### **Support Contact**
For further assistance, feel free to reach us at:
📧 **[support@apitradebuilder.com](mailto:support@apitradebuilder.com)**

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.
