"""
# Requeriments
yfinance
pandas
finta
numpy
google-cloud-bigquery
google-cloud-storage
pyarrow
lxml
"""


import yfinance as yf
import pandas as pd
import datetime
from finta import TA
from google.cloud import bigquery
import pyarrow



def get_data(event, context):
     # top 50 S&P Stocks
     stocks = ["MSFT", "AAPL", "AMZN", "GOOG", "FB", "JNJ", "WMT", "V", "PG", "JPM",
              "UNH", "MA", "INTC", "VZ", "HD", "T", "MRK", "KO", "PFE", "BAC", "DIS", "PEP",
              "NFLX", "XOM", "CSCO", "NVDA", "CMCSA", "ORCL", "ABT", "ADBE", "CVX", "LLY", "CRM",
              "COST", "NKE", "AES", "MDT", "MCD", "AMGN", "BMY", "PYPL", "TMO", "ABBV",
              "PM", "NEE", "CHTR", "WFC", "ACN", "LMT"]
     data_frame = pd.DataFrame()
     for stock in stocks:
          start_time = (datetime.date.today() - datetime.timedelta(365)).isoformat()
          end_time = datetime.datetime.today()
          data = yf.download(stock, start=start_time, end=end_time, auto_adjust=True, progress=True)
          # S&P500 price and relative price difference.
          sp = yf.download('SPY', start=start_time, end=end_time, auto_adjust=True, progress=False)
          sp.columns = [x.lower() for x in sp.columns]
          sp = sp.add_prefix('sp_')
          sp['sp_percent_change'] = sp['sp_close'].pct_change(periods=1).astype(float)
          data = data.merge(sp, left_index=True, right_index=True)
          # Return
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
          # Some indicators need longer time so we can just remove first 500
          data.dropna(axis=0, inplace=True)

          # Append only the last result
          data_frame = pd.concat([data_frame, data.tail(1)])

     data_frame.reset_index(inplace=True, drop=True)
     # Include stock info
     url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
     stock_info_df = pd.read_html(url)[0]
     stock_info_df.rename(columns={'Symbol': 'ticker', 
                                   'Security': 'long_name', 
                                   'GICS Sector': 'sector', 
                                   'GICS Sub Industry': 'industry'},
                         inplace=True)
     stock_info_df = stock_info_df[['ticker', 'long_name', 'sector', 'industry']]
     df_merged = pd.merge(data_frame, stock_info_df, how='left', on='ticker')
     df_merged.columns
     df_merged.head()

     # Add to bigquery
     client = bigquery.Client(project='tradebot-tfm')

     dataset_id = 'tradingbot_query'
     table_id = 'dataset_hist'

     dataset_ref = client.dataset(dataset_id)
     table_ref = dataset_ref.table(table_id)

     job_config = bigquery.LoadJobConfig()
     job_config.source_format = 'CSV'
     job_config.autodetect = True
     job_config.ignore_unknown_values = True
     job = client.load_table_from_dataframe(
     df_merged,
     table_ref,
     location='US',
     job_config=job_config
     )
     job.result()

     return 'Success'

