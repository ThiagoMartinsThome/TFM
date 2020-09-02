# TRADEBOT_TFM

## Link: https://tradebot-tfm.herokuapp.com/

• Definition: Trading bots are computer programs that use various indicators to recognize trends and automatically execute trades.


• Project Objectives:

  - Build a Bot that can predict the short-term stock movements (next 20 labor day trend) of the Top companies listed in the SP&500 index using Quantitative Finance and Machine Learning methods. For that I'm going to use Google Cloud Plataform to automate the process and as a BBDD,  Alpaca as a trade paper API to back test the performance and simulate a real market, and finally I will use Streamlit as a front end platform and Heroku for deploying.
  
• Project Structure:

  - ETL process (extract, transform, load):
    
    - Extract the data. "gcp_daily_data.py" to be impemented as a cloud function with a weekday scheduler and "get_data.py" for manual extraction: 
      1. This function extract the data of the SP&500 index and the top companies listed in (yfinance library).
      2. Feature extraction using the most common technical indicators (finta library).
      3. Calculate the target (short-term stock movements for next 20 labor days trend).
      4. Scrapp the company information like: long_name, sector and industry.
      5. Save a train dataset and a historical dataset (do not include the target values) as csv file.
      6. Load the historical dataset in the google cloud platform (Cloud Storage / BigQuery).
      
  - Model creation and predictions:
  
    - Define the best model to predict the target. Notebooks and saved models placed in the models folder :
      1. Build a pipeline to scale the dataset.
      2. Machine Learning models:
        - ExtraTreeRegressor
        - RandomForestRegressor
      3. Deep Learning models:
        - Bidirectional LSTM model
        - ConvLSTM2D model
      4. Test and evaluate and save the best model
      5. Load the model in the google cloud platform (CloudStorage)
      
    - Create a function in the cloud to predict daily. "gcp_get_predictions.py"
      1. Load the historical data from BigQuery.
      2. Load the model from Cloud Storage.
      3. Predict
      4. Insert the results back to BigQuery in the predictions table
      
  - TradeBot:
  
    - Create a function in the cloud to perform the bot daily. "tradebot.py"
      1. Get the Alpaca keys from cloud storage and initialize the API.
      2. Get the current portfolio.
      3. Get the top stocks predicted (strong_buy and buy labels) from the Bigquery predictions table and apply the portfolio optimization using PyPortfolioOpt library.
      4. Define the stocks to buy and to sell.
      5. Send a buy/sell order to the API.
      6. Update the strategy_log table with the actions taken.
      
  - Front-end Visualization. Streamlit app.
    
    -  Project description: Short description of the project.
      
    -  Market Status: Check if the market is open.
    
    - Technical Analysis Viewer:
      1. Stock selection.
      2. Select the date range
      3. Technical Analysis:
        - Company Name
        - Sector
        - Industry
        - Website
        - OHLCV table
        - Graphic Analysis (RSI, Candlestick, MACD, Volume)
        
    - Portfolio Management:
      1. Porfolio information:
        - Buying power
        - Cash
        - Porfolio value
        - Balance
        - Total profit/loss
       2. Equity profit/loss history
       3. Current portfolio table
    
    - Get Predictions:
      1. Stocks predictions table with the percentage earnings expected for the next 20 days.
      2. Maximum portfolio size selection range
      3. Cash input selection
      4. Buy/Sell recommendations.
      5. Submit a buy/sell order
      
    - Portfolio history:
      1. Get historical actions taken by the bot
      
  - Deploy the project online:
    
    - Heroku:
      1. Create a a setup.sh and a profile files to connect with streamlit.
      2. Set config variables to store the google cloud credentials.
      3. Deploy the model throughout github.
      
  
        
      
• Technical indicators used:

RSI - Compares the magnitude of recent gains and losses over a specified time period to measure speed and change of price movements of a security. It is primarily used to attempt to identify overbought or oversold conditions in the trading of an asset.

STOCH - The stochastic oscillator presents the location of the closing price of a stock in relation to the high and low range of the price of a stock over a period of time, typically a 14-day period.

CCI - Measures the difference between a security’s price change and its average price change. High positive readings indicate that prices are well above their average, which is a show of strength. Low negative readings indicate that prices are well below their average, which is a show of weakness.

ADX / DMI - Average Directional Movement Index (ADX).The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI) are derived from smoothed averages of these differences, and measure trend direction over time. These two indicators are often referred to collectively as the Directional Movement Indicator (DMI).The Average Directional Index (ADX) is in turn derived from the smoothed averages of the difference between +DI and -DI, and measures the strength of the trend (regardless of direction) over time. Using these three indicators together, chartists can determine both the direction and strength of the trend.

AO - The Awesome Oscillator is an indicator used to measure market momentum. AO calculates the difference of a 34 Period and 5 Period Simple Moving Averages. The Simple Moving Averages that are used are not calculated using closing price but rather each bar’s midpoints. AO is generally used to affirm trends or to anticipate possible reversals.

MOM - Market Momentum.

MACD - Moving Average Convergence Divergence. Is a trend-following momentum indicator that shows the relationship between two moving averages of prices.

STOCHRSI -  Stochastic RSI.

WILLIAMS - Developed by Larry Williams, Williams %R is a momentum indicator that is the inverse of the Fast Stochastic Oscillator. Also referred to as %R, Williams %R reflects the level of the close relative to the highest high for the look-back period. In contrast, the Stochastic Oscillator reflects the level of the close relative to the lowest low. %R corrects for the inversion by multiplying the raw value by -100. As a result, the Fast Stochastic Oscillator and Williams %R produce the exact same lines, only the scaling is different. Williams %R oscillates from 0 to -100.

EBBP - Bull power and Bear Power.

UO - Ultimate Oscillator .Larry Williams’ (1976) signal, a momentum oscillator designed to capture momentum across three different timeframes.

VAMA - Volume Adjusted Moving Average.

HMA - Hull Moving Average.

ICHIMOKU - It identifies the trend and look for potential signals within that trend.
SMM - Simple Moving Median.

SSMA - Smoothed Moving Average.

SMA - Simple Moving Average.

EMA - Exponential Moving Average.

TEMA - Triple Exponential Moving Average.

TRIMA -  Triangular Moving Average.

TRIX - Shows the percent rate of change of a triple exponentially smoothed moving average.

ER - Kaufman Efficiency Indicator.

KAMA - Kaufman’s Adaptive Moving Average (KAMA). Moving average designed to account for market noise or volatility. KAMA will closely follow prices when the price swings are relatively small and the noise is low. KAMA will adjust when the price swings widen and follow prices from a greater distance. This trend-following indicator can be used to identify the overall trend, time turning points and filter price movements.

ZLEMA - Zero Lag Exponential Moving Average.

WMA - Weighted Moving Average.

EVWMA - Elastic Volume Moving Average.

VWAP - Volume Weighted Average Price is equals the dollar value of all trading periods divided by the total trading volume for the current day. The calculation starts when trading opens and ends when it closes. Because it is good for the current trading day only, intraday periods and data are used in the calculation.

PPO - Percentage Price Oscillator .

ROC - The Rate-of-Change (ROC) indicator, which is also referred to as simply Momentum, is a pure momentum oscillator that measures the percent change in price from one period to the next. The ROC calculation compares the current price with the price “n” periods ago. The plot forms an oscillator that fluctuates above and below the zero line as the Rate-of-Change moves from positive to negative. As a momentum oscillator, ROC signals include centerline crossovers, divergences and overbought-oversold readings. Divergences fail to foreshadow reversals more often than not, so this article will forgo a detailed discussion on them. Even though centerline crossovers are prone to whipsaw, especially short-term, these crossovers can be used to identify the overall trend. Identifying overbought or oversold extremes comes naturally to the Rate-of-Change oscillator.
  
    
