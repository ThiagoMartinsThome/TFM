import streamlit as st
import numpy as np
import pandas as pd
import datetime
from datetime import timezone
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from finta import TA as ta
import tradebot as tb
from PIL import Image

stock_list = ["MSFT", "AAPL", "AMZN", "GOOG", "FB", "JNJ", "WMT", "V", "PG", "JPM",
              "UNH", "MA", "INTC", "VZ", "HD", "T", "MRK", "KO", "PFE", "BAC", "DIS", "PEP",
              "NFLX", "XOM", "CSCO", "NVDA", "CMCSA", "ORCL", "ABT", "ADBE", "CVX", "LLY", "CRM",
              "COST", "NKE", "AES", "MDT", "MCD", "AMGN", "BMY", "PYPL", "TMO", "ABBV",
              "PM", "NEE", "CHTR", "WFC", "ACN", "LMT"]

st.markdown('# *Trade Bot WebApp*')
st.markdown('by Thiago Martins')
st.image('stock_image.gif', use_column_width=True)


# Project Title
st.sidebar.markdown('# *TradeBot*')
if st.sidebar.checkbox("Project description"):
    """
    Definition: Trade bots are computer programs that use various indicators to 
    recognize trends and automatically execute trades.

    Objectives: Predict the short-term stock movements (next 20 labor day trend)
    of the top companies listed in the SP&500 index using Quantitative Finance 
    and Machine Learning methods.
    
    Project Content:
        
        1. Market Status: Check if the market is open.
        
        2. Technical Analysis Viewer:
            - Stock selection.
            - Select the date range
            - Technical Analysis:
                Company Name
                Sector
                Industry
                Website
                OHLCV table
                Graphic Analysis (RSI, Candlestick, MACD, Volume)
        3. Portfolio Management:
            - Portfolio information:
                Buying power
                Cash
                Portfolio value
                Balance
                Total profit/loss
                Equity profit/loss history
                Current portfolio table
        4. Get Predictions:
            - Stocks predictions table with the percentage earnings expected for 
            the next 20 days.
            - Maximum portfolio size selection range
            - Cash input selection
            - Buy/Sell recommendations.
            - Submit a buy/sell order
        5. Portfolio history:
            - Get historical actions taken by the bot
        
    """



# Check if the market is open
clock, market_status, message = tb.get_market_status()
# Calculate remaining hours until market opens
openingTime = clock.next_open.replace(tzinfo=timezone.utc).timestamp()
currTime = clock.timestamp.replace(tzinfo=timezone.utc).timestamp()
sec = int(openingTime - currTime)
hours = sec // 3600
minutes = sec // 60 - hours * 60
timeToOpen = "%d:%02d" % (hours, minutes)

if market_status == 1:
    st.sidebar.markdown("## *Market Status: *" + str(message))
else:
    st.sidebar.markdown("## *Market Status: *" + str(message))
    st.sidebar.markdown("##### *Wait " + str(timeToOpen) + " hs until market open.*")



# TA Analysis  ########

if st.sidebar.checkbox("Technical Analysis Viewer"):
    st.header('**Technical Analysis**')
    stock = st.sidebar.multiselect('Choose one stock', stock_list, ["AAPL"])

    @st.cache()
    def get_sp500_info():
        sp500_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_table = sp500_list[0]
        sp500_table.rename(columns={'Symbol': 'ticker',
                                    'Security': 'long_name',
                                    'GICS Sector': 'sector',
                                    'GICS Sub Industry': 'industry'}, inplace=True)
        sp500_table = sp500_table[['ticker', 'long_name', 'sector', 'industry']]
        sp500_table.set_index('ticker', inplace=True)
        return sp500_table
    sp500_table = get_sp500_info()

    if len(stock) > 1:
        st.error('You must choose only one company...')

    else:
        for item in stock:
            try:
                serie = sp500_table.loc[item]
                sector = serie['sector']
                longName = serie['long_name']
                industry = serie['industry']
                try:
                    ticker = yf.Ticker(item)
                    info_dict = ticker.info
                    website = info_dict.get('website')

                except:

                    website = "Not available"
                st.write('**Company name:**', longName)
                st.write('**Sector:**', sector)
                st.write('**Industry:**', industry)
                st.write('**Website:**', website)




                # Set 2 years range by default
                start_time = st.sidebar.date_input("Start Date", (datetime.date.today() - datetime.timedelta(365*2)))
                end_time = st.sidebar.date_input("End Date", datetime.datetime.today())

                def load_data(stock, start_time, end_time):
                    # Load data and
                    df = yf.download(stock, start=start_time, end=end_time, auto_adjust=False, progress=False)
                    lowercase = lambda x: str(x).lower()
                    df.rename(lowercase, axis='columns', inplace=True)

                    # Create technical analysis features
                    def appendData(maindf, dataarray, namesarray=None):
                        if namesarray == None:
                            return maindf.join(pd.DataFrame(dataarray), how='outer')
                        return maindf.join(pd.DataFrame(dataarray, columns=namesarray), how='outer')

                    # Oscillators
                    # RSI
                    df = appendData(df, ta.RSI(df))
                    # MACD
                    df = appendData(df, ta.MACD(df)).rename(columns={"SIGNAL": "MACD_SIGNAL"})
                    df["MACD_HIST"] = df['MACD'].diff()
                    # Moving Averages
                    sma_averages = [10, 30]
                    # SMA, EMA
                    for i in sma_averages:
                        df = appendData(df, ta.SMA(df, i))
                    df.reset_index(inplace=True)
                    return df

                # Technical Analysis table
                data = load_data(stock, start_time, end_time)
                data_table = data.sort_values(by='Date', ascending=False)
                data_table.reset_index(drop=True, inplace=True)
                data_table.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
                                  inplace=True)
                st.dataframe(data_table[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].style.set_precision(2))

                # Plot
                def plot_data(data):
                    fig = make_subplots(rows=4, cols=1,
                                        #subplot_titles=("Candlesticks", "MACD", "RSI", "Volume (Million)"),
                                        #row_titles=("RSI", "", "MACD", "Volume (Million)"),
                                        shared_xaxes=True,
                                        vertical_spacing=0.0, row_heights=[0.2, 0.7, 0.3, 0.2],
                                        specs=[[{"secondary_y": False}],
                                               [{"secondary_y": False}],
                                               [{"secondary_y": True}],
                                               [{"secondary_y": False}]],
                                        print_grid=False
                                        )
                    # Plot RSI
                    # Above 70% = overbought, below 30% = oversold
                    fig.add_trace(go.Scatter(x=data.Date,
                                             y=data['14 period RSI'], mode='lines',
                                             name="RSI",
                                             line=dict(color='royalblue', width=1.5)),
                                  secondary_y=False,
                                  row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.Date,
                                             y=[70] * len(data.index),
                                             mode="lines",
                                             line=go.scatter.Line(color="LightSteelBlue", width=1, dash='dash'),
                                             showlegend=False), secondary_y=False, row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.Date,
                                             y=[30] * len(data.index),
                                             mode="lines",
                                             line=go.scatter.Line(color="LightSteelBlue", width=1, dash='dash'),
                                             showlegend=False), secondary_y=False, row=1, col=1)
                    fig.update_yaxes(title_text="<b>RSI</b>", secondary_y=False, row=1, col=1)
                    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='gainsboro')
                    fig.update_yaxes(showgrid=False, zeroline=False, showline=False)
                    # Plot candlestick chart
                    fig.add_trace(go.Candlestick(x=data.Date,
                                                 open=data['Open'],
                                                 high=data['High'],
                                                 low=data['Low'],
                                                 close=data['Close'], name="Candelstick"), secondary_y=False, row=2, col=1)
                    fig.update_xaxes(rangeslider_visible=False)
                    fig.add_trace(go.Scatter(x=data.Date,
                                             y=data['30 period SMA'], mode='lines', name='30 SMA', line=dict(color='crimson', width=1)),
                                  secondary_y=False,
                                  row=2, col=1)
                    fig.add_trace(go.Scatter(x=data.Date,
                                             y=data['10 period SMA'], mode='lines', name='10 SMA', line=dict(color='springgreen', width=1)),
                                  secondary_y=False,
                                  row=2, col=1)

                    # Plot MACD
                    fig.add_trace(go.Scatter(x=data.Date,
                                             y=data["MACD"], mode='lines', name="MACD", line=dict(color='royalblue', width=1)),
                                  secondary_y=False, row=3, col=1, )
                    fig.add_trace(go.Scatter(x=data.Date,
                                             y=data["MACD_SIGNAL"], mode='lines', name="MACD_SIGNAL", line=dict(color='orange', width=1)),
                                  secondary_y=False, row=3, col=1)
                    fig.add_trace(go.Bar(x=data.Date,
                                         y=data["MACD_HIST"], name="MACD_HIST", marker_color='Purple'),
                                  secondary_y=True, row=3, col=1)
                    fig.update_yaxes(title_text="<b>MACD</b>", secondary_y=False, row=3, col=1)
                    # ax_macd.bar(data.index, data["macd_hist"] * 3, label="hist")

                    # Show volume in millions
                    fig.add_trace(go.Bar(x=data.Date,
                                         y=data["Volume"] / 1000000, marker=dict(color="CornflowerBlue"),
                                         name="Volume(K)"), secondary_y=False, row=4, col=1)
                    fig.update_yaxes(title_text="<b>Vol.(K)</b>", secondary_y=False, row=4, col=1)

                    # Range Selector
                    fig.update_xaxes(
                        rangeslider_visible=False, row=1,
                        rangebreaks=[dict(bounds=["sat", "mon"])],
                        type="date",
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")])))

                    #go.Layout()
                    fig.update_layout(xaxis_rangeslider_visible=False,
                                      autosize=True,
                                      legend_orientation="h",
                                      overwrite=True,
                                      font=dict(size=8))
                    # ggplot2, plotly_dark, seaborn, plotly, plotly_white, presentation, or xgridoff
                    fig.layout.template = "presentation"

                    return fig
                fig = plot_data(data_table)
                st.plotly_chart(fig, use_container_width=True)

            except:
                my_placeholder = st.empty()
                my_placeholder.text("Choose one stock to continue")


# Get Alpaca positions and current portfolio value ####################################
# Portfolio Management checkbox
if st.sidebar.checkbox("Portfolio Management"):

    st.header('**Portfolio Management**')
    df_pf, portfolio_value, buying_power, cash, pf_hist_df, equity = tb.get_positions()

    st.write('**Buying_power: **', round(buying_power), "USD")
    st.write('**Cash: **', round(cash), "USD")
    st.write('**Porfolio Value: **', round(portfolio_value), "USD")
    st.write('**Balance: **', round(equity), "USD")
    st.write('**Total Profit/Loss: **', round(pf_hist_df.profit_loss_pct[-1]*100, 2), "%")

    fig2 = make_subplots(rows=1, cols=1,
                         print_grid=False)
    fig2.add_trace(go.Scatter(x=pf_hist_df.index,
                              y=pf_hist_df.equity, mode='lines',
                              line=dict(color='royalblue', width=4)), row=1, col=1)
    fig2.update_layout(title="<b>Equity Profit/Loss</b>", width=700, height=300)
    st.plotly_chart(fig2)

    st.subheader("Current Portfolio")

    def color(val):
        if val > 0:
            color = 'green'
        elif val < 0:
            color = 'red'
        else:
            color = 'black'
        return 'color: {}'.format(color)

    pf_table = df_pf.rename(columns={'ticker': 'Stock',
                                     'price': 'Price',
                                     'quantity': 'Quantity',
                                     'market_value': 'Market Value',
                                     'total_profit_USD': 'Total Profit $'
                                     })

    pf_table.sort_values(by=['Quantity', 'Stock'], ascending=False, inplace=True, ignore_index=True)
    st.dataframe(pf_table.style.applymap(color, subset=['Total Profit $']).set_precision(2))

    # Highlight Dataframe recommendation
    def color_rec(val):
        if val == 'Strong Buy':
            color = 'green'
        elif val == 'Buy':
            color = 'Seagreen'
        elif val == 'Neutral':
            color = 'grey'
        elif val == 'Sell':
            color = 'tomato'
        elif val == 'Strong Sell':
            color = 'red'
        return 'color: {}'.format(color)


    # Get predictions ############################################################################
    if st.sidebar.checkbox("Get Predictions"):
        # Predictions table
        @st.cache()
        def get_predictions():
            df, df_pred, current_data_date = tb.get_predictions()
            return df, df_pred, current_data_date
        df, df_pred, current_data_date = get_predictions()

        st.subheader("Stocks predictions: % earnings expected for the next 20 days")
        pred_table = df_pred[["ticker", "long_name", "predictions", "recommendations"]]
        pred_table.sort_values(by='predictions', ascending=False, inplace=True)
        pred_table = pred_table.rename(columns={'ticker': 'Stock',
                                                'long_name': 'Company',
                                                'recommendations': 'Recommendations',
                                                'predictions': '%_chg_20_days'})
        pred_table.reset_index(inplace=True, drop=True)

        st.write(pred_table.style.set_precision(2).applymap(color_rec, 'Recommendations'))

        df_top_stocks = df_pred[(df_pred['recommendations'] == 'Strong Buy') | (df_pred['recommendations'] == 'Buy')]
        max_rec_pf = len(df_top_stocks)

        # Set the max portfolio size we want
        portfolio_size = st.sidebar.slider("Max. Portfolio Size: ", 2, max_rec_pf, max_rec_pf, 1, format=None)
        input_cash = st.sidebar.slider("Cash Amount: ", 1000.0,
                                       buying_power, cash,
                                       format=None)

        # Table with the portfolio to sell and to buy
        # Call the function
        # Top Stocks

        df_buy, sell_list = tb.get_top_stocks(
            df=df,
            df_pred=df_pred,
            date=current_data_date,
            cash=input_cash,
            df_pf=df_pf,
            pf_size=portfolio_size
        )

        # Get the current prices and the number of shares to sell"""
        df_sell = tb.sell_stocks(
            df=df,
            df_pf=df_pf,
            sell_list=sell_list,
            date=current_data_date
        )

        # List of stocks we need to sell
        df_sell_final = tb.stock_diffs(
            df_sell=df_sell,
            df_pf=df_pf,
            df_buy=df_buy
        )

        # List of stocks we need to buy
        df_buy_new = tb.df_buy_new(
            df_pf=df_pf,
            df_buy=df_buy
        )

        st.header('**Tradebot Portfolio Recommendation**')

        st.subheader('Stocks you should buy !!')
        table_buy_new = df_buy_new.rename(columns={'ticker': 'Stock',
                                                'quantity': 'Quantity',
                                                'close': 'Close'})
        st.write(table_buy_new[['Stock', 'Quantity', 'Close']].style.set_precision(2))

        if df_sell is not None:
            st.subheader('Stocks you should sell !!')
            table_sell = df_sell.rename(columns={'ticker': 'Stock',
                                                    'quantity': 'Quantity',
                                                    'close': 'Close'})
            st.write(table_sell[['Stock', 'Quantity', 'Close']].style.set_precision(2))
        else:
            st.subheader('No recommended stocks to sell!!')
        ####################################################################

        if market_status == 1:
            if st.sidebar.button("Submit a buy order"):
                st.write(tb.sell_order(df_sell_final=df_sell_final))
                st.write(tb.buy_order(df_buy_new=df_buy_new))
                position_df = tb.strategy_log(df_buy_new, df_sell_final)
        elif market_status == 0:
            st.sidebar.button("Wait market to open")

        if st.sidebar.checkbox("Portfolio history"):
            @st.cache()
            def get_pf_hist():
                df_pf_hist = tb.get_pf_hist()
                return df_pf_hist

            table_pf_hist = get_pf_hist()
            st.header('**Portfolio History**')

            def color_action(val):
                if val == 'buy':
                    color = 'green'
                else:
                    color = 'red'
                return 'color: {}'.format(color)
            st.write(table_pf_hist.style.set_precision(2).applymap(color_action, 'Action'))

