import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import math
from finta import TA


def get_hist_data(rel_chg_period=20):
    # top 50 S&P Stocks
    stocks = ["MSFT", "AAPL", "AMZN", "GOOG", "FB", "JNJ", "WMT", "V", "PG", "JPM",
              "UNH", "MA", "INTC", "VZ", "HD", "T", "MRK", "KO", "PFE", "BAC", "DIS", "PEP",
              "NFLX", "XOM", "CSCO", "NVDA", "CMCSA", "ORCL", "ABT", "ADBE", "CVX", "LLY", "CRM",
              "COST", "NKE", "AES", "MDT", "MCD", "AMGN", "BMY", "PYPL", "TMO", "ABBV",
              "PM", "NEE", "CHTR", "WFC", "ACN", "LMT"]

    data_frames = pd.DataFrame()
    for stock in stocks:
        print(stock, "Gathering")
        # Historical 500 months data
        start_time = (datetime.date.today() - datetime.timedelta(500 * 365 / 12)).isoformat()
        # start_time = (datetime.date.today() - datetime.timedelta(365)).isoformat()
        end_time = datetime.datetime.today()
        data = yf.download(stock, start=start_time, end=end_time, auto_adjust=True, progress=True)
        # S&P500 price and relative price difference.
        sp = yf.download('SPY', start=start_time, end=end_time, auto_adjust=True, progress=False)
        sp.columns = [x.lower() for x in sp.columns]
        sp = sp.add_prefix('sp_')
        sp['sp_percent_change'] = sp['sp_close'].pct_change(periods=1).astype(float)
        data = data.merge(sp, left_index=True, right_index=True)
        # Returns
        data['percent_change'] = data['Close'].pct_change(periods=1).astype(float)
        # Daily percent change as compared to the S&P500
        data['relative_change'] = data['percent_change'] - data['sp_percent_change']
        data.reset_index(inplace=True)
        # Split the date
        data['year'] = data.Date.dt.year.astype('float64')
        data['month'] = data.Date.dt.month.astype('float64')
        data['day'] = data.Date.dt.day.astype('float64')
        data['day_of_year'] = data.Date.dt.dayofyear.astype('float64')
        data['week_of_year'] = data.Date.dt.isocalendar().week.astype('float64')
        data['quarter'] = data.Date.dt.quarter.astype('float64')

        data.columns = [x.lower() for x in data.columns]

        # Add the financial indicators
        indicators = ['RSI', 'STOCH', 'CCI', 'ADX', 'DMI', 'AO', 'MOM', 'MACD', 'STOCHRSI', 'WILLIAMS', 'EBBP', 'UO',
                      'VAMA', 'HMA', 'ICHIMOKU', 'SMM', 'SSMA', 'DEMA', 'TEMA', 'TRIMA', 'TRIX', 'ER', 'KAMA', 'ZLEMA',
                      'WMA', 'EVWMA', 'VWAP', 'PPO', 'ROC']

        for indicator in indicators:
            df = None
            df = eval('TA.' + indicator + '(data)')
            if not isinstance(df, pd.DataFrame):
                df = df.to_frame()
            # FinTA does not name this column
            if indicator == 'UO':
                df.columns = ['UO']
            data = data.merge(df, left_index=True, right_index=True)
        # Ichimoku -- Base (Kijun) and Conversion (Tenkan) Only
        data.drop(columns=['senkou_span_a', 'SENKOU', 'CHIKOU'], axis=1, inplace=True)

        # Add moving averages
        averages = [5, 10, 20, 30, 50, 100, 200]
        for avg in averages:
            sma = TA.SMA(data, avg)
            sma = sma.to_frame()
            data = data.merge(sma, left_index=True, right_index=True)
            ema = TA.EMA(data, avg)
            ema = ema.to_frame()
            data = data.merge(ema, left_index=True, right_index=True)

        # Target relative change in n days
        # Comment this block to generate only train data
        for index, row in data.iterrows():
            # Sums the total relative change compared to S&P over
            percent_change = data.loc[index + 1: index + rel_chg_period]['relative_change'].sum() * 100
            # Need for our model
            if math.isnan(percent_change):
                percent_change = 0
            else:
                percent_change = int(round(percent_change))
            data.loc[index, 'short_result'] = percent_change
        # input ticker column and place in the first position
        data['ticker'] = stock
        col = data.pop("ticker")
        data.insert(0, "ticker", col)

        # Fix labels
        data.columns = data.columns.str.replace(' ', '_')
        data.columns = data.columns.str.replace('+', 'plus')
        data.columns = data.columns.str.replace('-', 'minus')
        data.columns = data.columns.str.replace('%', '')
        data.columns = data.columns.str.replace('.', '')
        # BigQuery do not accept col names with numbers or "_"
        for col in data.columns:
            if col.startswith(tuple('0123456789')):
                data.rename(columns={col: 'f_{}'.format(col)}, inplace=True)
        data.columns = [x.lower() for x in data.columns]
        # Remove NAN
        data.dropna(axis=0, inplace=True)
        # Give a response to the user
        if data.isnull().sum().sum() > 0:
            print(stock, "NULL")
        print(stock, "Success")
        # Append data
        data_frames = pd.concat([data_frames, data])
        # data_frames = pd.concat([data_frames, data.tail(1)])

    data_frames.reset_index(inplace=True, drop=True)
    # Include stock info
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    stock_info_df = pd.read_html(url)[0]
    stock_info_df.rename(columns={'Symbol': 'ticker',
                                  'Security': 'long_name',
                                  'GICS Sector': 'sector',
                                  'GICS Sub Industry': 'industry'},
                         inplace=True)
    stock_info_df = stock_info_df[['ticker', 'long_name', 'sector', 'industry']]
    df_merged = pd.merge(data_frames, stock_info_df, how='left', on='ticker')
    # Save data in a csv format
    df_merged.to_csv('data/data_20day_chg_target.csv', index=False)
    df_merged.drop(columns='short_result', axis=1, inplace=True)
    df_merged.to_csv('data/data_20day_chg.csv', index=False)

    return df_merged


df = get_hist_data()

print(df.isnull().sum().sum())
print(df.columns)
print(df)
print(df.info())
print(df.describe())
print(len(np.unique(df['ticker'])))



