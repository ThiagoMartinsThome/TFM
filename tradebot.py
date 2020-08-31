"""#### Trading Bot ####"""
#https://us-central1-tradebot-tfm.cloudfunctions.net/predict

"""# Libraries"""
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
import alpaca_trade_api as tradeapi
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import pytz
from datetime import datetime
import os


credentials_path = "/Users/thiago/PycharmProjects/tradebot_tfm/keys/tradebot-tfm.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

"""# Get keys"""
# Get the Alpaca api keys from cloud storage
storage_client = storage.Client(project='tradebot-tfm')
bucket = storage_client.get_bucket('tradebot_bucket')
blob = bucket.blob('alpaca_keys.txt')
api_key = blob.download_as_string()
secret_key = api_key.splitlines()[0].decode()
key_id = api_key.splitlines()[1].decode()
alpha_key = api_key.splitlines()[2].decode()

"""# Initialize the alpaca api """
base_url = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(key_id=key_id, secret_key=secret_key, base_url=base_url, api_version='v2')


def get_market_status():
    """# Check if market is open"""
    clock = api.get_clock()
    if clock.is_open:
        market_status = 1
        return clock, market_status, "OPENED!!!"
    else:
        market_status = 0
        return clock, market_status, "CLOSED!!!"


def get_positions(api=api):
    """# Get Alpaca positions"""
    # Get the current positions from alpaca and create a df
    positions = api.list_positions()
    info_account = api.get_account()
    buying_power = info_account.buying_power
    cash = info_account.cash
    equity = info_account.equity


    symbol, price, qty, market_value, total_profit = [], [], [], [], []

    for each in positions:
        symbol.append(each.symbol)
        price.append(float(each.current_price))
        qty.append(int(each.qty))
        market_value.append(float(each.market_value))
        total_profit.append(float(each.unrealized_pl))


    df_pf = pd.DataFrame({
        'ticker': symbol,
        'price': price,
        'quantity': qty,
        'market_value': market_value,
        'total_profit_USD': total_profit})

    df_pf.sort_values(by=['quantity'], ascending=False, inplace=True)
    df_pf.reset_index(inplace=True, drop=True)

    # Current portfolio value
    portfolio_value = round(df_pf['market_value'].sum(), 2)

    pf_hist = api.get_portfolio_history()

    pf_hist_df = pd.DataFrame({
        'timestamp': pf_hist.timestamp,
        'equity': pf_hist.equity,
        'profit_loss': pf_hist.profit_loss,
        'profit_loss_pct': pf_hist.profit_loss_pct,

    })

    pf_hist_df['date'] = pf_hist_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
    pf_hist_df.drop('timestamp', axis=1, inplace=True)
    pf_hist_df.set_index('date', inplace=True, drop=True)

    return df_pf, portfolio_value, float(buying_power), float(cash), pf_hist_df, equity


def get_predictions():
    # BQ credentials
    client = bigquery.Client(project='tradebot-tfm')

    ### df
    # Load the historical stock data from BQ
    sql_hist = """
        SELECT *
        FROM `tradebot-tfm.tradingbot_query.dataset_hist`
        WHERE
            date >= DATE_SUB(CURRENT_DATE(), INTERVAL 22 DAY)
        ORDER BY
            date DESC,
            ticker
        """

    df = client.query(sql_hist).to_dataframe()

    # Convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Get the latest date for the data we have
    current_data_date = df['date'].max()

    ## df_pred
    # Load the historical stock data from BQ
    sql_pred = """
        SELECT *
        FROM `tradebot-tfm.tradingbot_query.predictions`
        WHERE
            date >= DATE_SUB(CURRENT_DATE(), INTERVAL 4 DAY)
        ORDER BY
            date DESC,
            recommendations
        """

    df_pred = client.query(sql_pred).to_dataframe()

    # Convert the date column to datetime
    df_pred['date'] = pd.to_datetime(df_pred['date'])
    df_pred = df_pred[df_pred['date'] == current_data_date]
    return df, df_pred, current_data_date



# Function to get the momentum stocks we want
def get_top_stocks(df, df_pred, date, cash, df_pf, pf_size=20):
    # Filter the df to get the top n stocks for the latest day
    df_top_stocks = df_pred[(df_pred['recommendations'] == 'Strong Buy') | (df_pred['recommendations'] == 'Buy')]
    df_top_stocks = df_top_stocks.loc[df['date'] == pd.to_datetime(date)]
    df_top_stocks = df_top_stocks.sort_values(by='predictions', ascending=False).head(pf_size)

    # Set the universe to the top momentum stocks for the period
    universe = df_top_stocks['ticker'].tolist()

    # Create a df with just the stocks from the universe
    df_u = df.loc[df['ticker'].isin(universe)]

    # Create the portfolio
    # Pivot to format for the optimization library
    df_u = df_u.pivot_table(
        index='date',
        columns='ticker',
        values='close',
        aggfunc='sum')

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df_u)
    S = risk_models.sample_cov(df_u)

    # Optimise the portfolio for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S, gamma=1)  # Use regularization (gamma=1)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # Allocate
    latest_prices = get_latest_prices(df_u)

    da = DiscreteAllocation(
        cleaned_weights,
        latest_prices,
        total_portfolio_value=cash)

    allocation = da.lp_portfolio()[0]
    # Put the stocks and the number of shares from the portfolio into a df
    symbol_list = []
    num_shares_list = []

    for symbol, num_shares in allocation.items():
        symbol_list.append(symbol)
        num_shares_list.append(num_shares)

    # Now that we have the stocks we want to buy we filter the df for those ones
    df_buy = df.loc[df['ticker'].isin(symbol_list)]
    # Filter for the period to get the closing price
    df_buy = df_buy.loc[df_buy['date'] == date].sort_values(by='ticker')
    # Add in the qty that was allocated to each stock
    df_buy['quantity'] = num_shares_list

    # Calculate the amount we own for each stock
    df_buy['amount_held'] = df_buy['close'] * df_buy['quantity']
    df_buy = df_buy.loc[df_buy['quantity'] != 0]

    # Create a list of stocks to sell based on what is currently in our pf
    sell_list = list(set(df_pf['ticker'].tolist()) - set(df_buy['ticker'].tolist()))

    return round(df_buy[['ticker', 'close', 'quantity', 'amount_held']], 2), sell_list





"""# Create a list of stocks to sell based on what is currently in our pf"""
def sell_stocks(df, df_pf, sell_list, date):
    """# Get the current prices and the number of shares to sell"""
    df_sell_price = df.loc[df['date'] == pd.to_datetime(date)]
    # Filter
    df_sell_price = df_sell_price.loc[df_sell_price['ticker'].isin(sell_list)]
    # Check to see if there are any stocks in the current ones to buy
    # that are not in the current portfolio. It's possible there may not be any
    if df_sell_price.shape[0] > 0:
        df_sell_price = df_sell_price[['ticker', 'close']]

        # Merge with the current pf to get the number of shares we bought initially
        # so we know how many to sell
        df_buy_shares = df_pf[['ticker', 'quantity']]
        df_sell = pd.merge(
            df_sell_price,
            df_buy_shares,
            on='ticker',
            how='left'
        )

        df_sell.sort_values(by='quantity', ascending=False, inplace=True)
        df_sell.reset_index(inplace=True, drop=True)
        df_sell = round(df_sell, 2)
    else:
        df_sell = None

    return df_sell


def stock_diffs(df_sell, df_pf, df_buy):
    """# Create a table with the stocks we need to buy and sell"""
    df_stocks_held_prev = df_pf[['ticker', 'quantity']]
    df_stocks_held_curr = df_buy[['ticker', 'quantity', 'close']]

    # Inner merge to get the stocks that are the same week to week
    df_stock_diff = pd.merge(
        df_stocks_held_curr,
        df_stocks_held_prev,
        on='ticker',
        how='inner'
    )

    # Check to make sure not all of the stocks are different compared to what we have in the pf
    if df_stock_diff.shape[0] > 0:
        # Calculate any difference in positions based on the new pf
        df_stock_diff['share_amt_change'] = df_stock_diff['quantity_x'] - df_stock_diff['quantity_y']

        # Create df with the share difference and current closing price
        df_stock_diff = df_stock_diff[[
            'ticker',
            'share_amt_change',
            'close'
        ]]

        # If there's less shares compared to last week for the stocks that
        # are still in our portfolio, sell those shares
        df_stock_diff_sale = df_stock_diff.loc[df_stock_diff['share_amt_change'] < 0]

        # If there are stocks whose qty decreased,
        # add the df with the stocks that dropped out of the pf
        if df_stock_diff_sale.shape[0] > 0:
            if df_sell is not None:
                df_sell_final = pd.concat([df_sell, df_stock_diff_sale], sort=True)
                # Fill in NaNs in the share amount change column with
                # the qty of the stocks no longer in the pf, then drop the qty columns
                df_sell_final['share_amt_change'] = df_sell_final['share_amt_change'].fillna(
                    df_sell_final['quantity'])
                df_sell_final = df_sell_final.drop(['quantity'], 1)
                # Turn the negative numbers into positive for the order
                df_sell_final['share_amt_change'] = np.abs(df_sell_final['share_amt_change'])
                df_sell_final.columns = df_sell_final.columns.str.replace('share_amt_change', 'quantity')
            else:
                df_sell_final = df_stock_diff_sale
                # Turn the negative numbers into positive for the order
                df_sell_final['share_amt_change'] = np.abs(df_sell_final['share_amt_change'])
                df_sell_final.columns = df_sell_final.columns.str.replace('share_amt_change', 'quantity')
        else:
            df_sell_final = None
    else:
        df_sell_final = df_sell #df_stocks_held_curr

    return df_sell_final


"""# Sell Order"""
def sell_order(df_sell_final):
    # Send the sell order to the api
    if df_sell_final is not None:
        symbol_list = df_sell_final['ticker'].tolist()
        qty_list = df_sell_final['quantity'].tolist()
        try:
            for symbol, qty in list(zip(symbol_list, qty_list)):
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
        except Exception:
            pass


# Buy the stocks that increased in shares compared
# to last week or any new stocks
def df_buy_new(df_pf, df_buy):
    # Left merge to get any new stocks or see if they changed qty
    df_buy_new = pd.merge(
        df_buy,
        df_pf,
        on='ticker',
        how='left'
    )

    # Get the qty we need to increase our positions by
    df_buy_new = df_buy_new.fillna(0)
    df_buy_new['quantity_new'] = df_buy_new['quantity_x'] - df_buy_new['quantity_y']

    # Filter for only shares that increased
    df_buy_new = df_buy_new.loc[df_buy_new['quantity_new'] > 0]
    if df_buy_new.shape[0] > 0:
        df_buy_new = df_buy_new[[
            'ticker',
            'quantity_new',
            'close'
        ]]
        df_buy_new = df_buy_new.rename(columns={'quantity_new': 'quantity'})
    else:
        df_buy_new = None

    return df_buy_new

"""# Buy Order"""
def buy_order(df_buy_new):
    # Send the buy order to the api
    if df_buy_new is not None:
        symbol_list = df_buy_new['ticker'].tolist()
        qty_list = df_buy_new['quantity'].tolist()
        try:
            for symbol, qty in list(zip(symbol_list, qty_list)):
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
        except Exception:
            pass



def strategy_log(df_buy_new, df_sell_final):

    df_buy_new['action'] = 'buy'
    if df_sell_final is not None:
        df_sell_final['action'] = 'sell'
        position_df = pd.concat([df_buy_new, df_sell_final], ignore_index=True)
    else:
        position_df = df_buy_new
    """# Log the updated pf
    positions = api.list_positions()

    symbol, qty, market_value = [], [], []

    for each in positions:
        symbol.append(each.symbol)
        qty.append(int(each.qty))
        market_value.append(float(each.market_value))

    position_df = pd.DataFrame({
        'ticker': symbol,
        'quantity': qty,
        'market_value': market_value})"""

    # Check if the market was open today
    today = datetime.today().astimezone(pytz.timezone("America/New_York"))
    today_fmt = today.strftime('%Y-%m-%d')

    # Add the current date and other info into the portfolio df for logging
    position_df['date'] = pd.to_datetime(today_fmt)
    position_df['strategy'] = 'Blstm_model'

    # Add the new pf to BQ
    # Format date to match schema
    position_df['date'] = position_df['date'].dt.date

    # Append it to the anomaly table
    client = bigquery.Client(project='tradebot-tfm')
    dataset_id = 'tradingbot_query'
    table_id = 'strategy_log'

    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.autodetect = True
    job_config.ignore_unknown_values = True

    job = client.load_table_from_dataframe(
        position_df,
        table_ref,
        location='US',
        job_config=job_config
    )

    job.result()
    return position_df

def get_pf_hist():
    # BQ credentials
    client = bigquery.Client(project='tradebot-tfm')
    # Load the historical stock data from BQ
    sql_hist = """
        SELECT
            date,
            ticker,
            quantity,
            market_value,
            strategy
        FROM `tradebot-tfm.tradingbot_query.strategy_log`
        ORDER BY
          date DESC
        """
    df_pf_hist = client.query(sql_hist).to_dataframe()

    # Convert the date column to datetime
    df_pf_hist['date'] = pd.to_datetime(df_pf_hist['date'])
    return df_pf_hist
