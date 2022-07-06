from math import floor
from sphinx.util import requests
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from datetime import date
from termcolor import colored as cl
pd.options.mode.chained_assignment = None

plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')
matplotlib.use("QT4Agg")

# account database
# account database code was written with the help of this link https://blog.jcharistech.com/2020/05/30/how-to-add-a-login-section-to-streamlit-blog-app/
import sqlite3

conn = sqlite3.connect("data.db")
c = conn.cursor()


def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username, password))
    conn.commit()


def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data


# needed code to remove the "made with Streamlit" and also the hamburger menu by https://www.youtube.com/watch?v=0_HlInz6HuM
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# side menu for the different pages
option = st.sidebar.selectbox("", {'Main Page', 'Stock Tweets', 'Trading Strategy', "Login", "Sign-Up"})
st.header(option)

if option == "Main Page":

    st.title("Stock Graphs and Statistics")

    START = "2020-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    # stock search
    g_tick = st.text_input("Enter Stock", value='AAPL', max_chars=5)


    # selected_stocks = st.selectbox("Select a stock ", stocks)

    # function for loading the stock data in the page
    @st.cache
    def get_API_data(ticker):
        stock_data = yf.download(ticker, START, TODAY)
        stock_data.reset_index(inplace=True)
        return stock_data

    stock_data = get_API_data(g_tick)


    # Plotting the data in the graph
    def plot_yf_data():
        fig = go.Figure(data=[go.Candlestick(x=stock_data['Date'],
                                             open=stock_data['Open'],
                                             high=stock_data['High'],
                                             low=stock_data['Low'],
                                             close=stock_data['Close'])])
        # adding a range slider to zoom in and out of the graph
        fig.update_layout(xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)


    plot_yf_data()

    st.subheader("Historical Data")
    st.write(stock_data)

    s_tick = st.text_input("Enter Ticker", value='AMZN', max_chars=5)
    # lnk = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{s_tick}.json")
    stock_tick_table = yf.Ticker(s_tick)


    # table to plot the statistics of chosen stocks by the user
    def plot_table():
        st.subheader("Stock Statistics")
        fig = go.Figure(data=go.Table(
            header=dict(values=["Market Cap", "Profit Margins", "EPS", "Dividend Yield", "Market Previous Close"]),

            cells=dict(values=[[stock_tick_table.info['marketCap']],
                               [format(stock_tick_table.info['profitMargins'] * 100, ".2f")+"%"],
                               [stock_tick_table.info["trailingEps"]],
                               [format(stock_tick_table.info["dividendYield"] )],
                               [stock_tick_table.info["regularMarketPreviousClose"]]]
                       )))
        st.plotly_chart(fig)


    plot_table()


# The code for the two trading strategies has been adapted from
# https://medium.com/codex/algorithmic-trading-with-macd-in-python-1c2769a6ad1b I have made sure to ask for
# permission to use this code and also get a student account for the Alpha Vantage API
if option == "Trading Strategy":

    strategy_option = st.selectbox('Please select a trading strategy', ('RSI', 'MACD'))

    if strategy_option == "MACD":

        MACD_description = st.write(
            "The MACD (Moving Average Convergence Divergence) measures the difference between two "
            "exponential moving averages and plots the "
            "difference as a line chart. The strategy is to buy – or close a short position – when "
            "the MACD crosses above the "
            "zero line, and sell – or close a long position – when the MACD crosses below the zero "
            "line. ")


        # MACD STRATEGY
        # IF MACD LINE > SIGNAL LINE => BUY THE STOCK
        # IF SIGNAL LINE > MACD LINE => SELL THE STOCK

        # store the data using the help of Alpha Vantage API to get all the necessary data
        def get_historical_dataMACD(symbol, start_date=None):
            # using the given API key to retrieve all the data
            api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&' \
                      f'apikey=4UTGJUQ9TWN4VJ0I&outputsize=full'
            raw_df = requests.get(api_url).json()
            df = pd.DataFrame(raw_df[f'Time Series (Daily)']).T
            df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
                                    '5. adjusted close': 'adj close', '6. volume': 'volume'})
            for i in df.columns:
                df[i] = df[i].astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.iloc[::-1].drop(['7. dividend amount', '8. split coefficient'], axis=1)
            if start_date:
                df = df[df.index >= start_date]
            return df


        # returns the historical data in a data frame
        api_tickMACD = st.text_input("Enter a stock", value="GOOGL", max_chars=5)
        api_stockMACD = get_historical_dataMACD(api_tickMACD, start_date="2017-01-01")
        # api_stock

        # in this function we calculate the macd, taking the prices of the stock and EMAs fast and slow
        # and also the period of the signal line. I first calculate the length of the fast-slow EMA using
        # the ewm function by pandas and I store it in the ema1 and ema2 var respectively.
        def get_macd(price, slow, fast, smooth):
            exp1 = price.ewm(span=fast, adjust=False).mean()
            exp2 = price.ewm(span=slow, adjust=False).mean()
            macd = pd.DataFrame(exp1 - exp2).rename(columns={'close': 'macd'})
            signal = pd.DataFrame(macd.ewm(span=smooth, adjust=False).mean()).rename(columns={'macd': 'signal'})
            hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns={0: 'hist'})
            frames = [macd, signal, hist]
            df = pd.concat(frames, join='inner', axis=1)
            return df


        stock_macd = get_macd(api_stockMACD['close'], 26, 12, 9)
        stock_macd.tail()
        # stock_macd

        # Plotting the MACD lines in the graph and later on using them and the prices to make trades
        # The MACD components are calculated in histograms on the bottom of the graph
        # The signal line covers the histograms of the MACD and gives a better display of the price moves
        def plot_macd(prices, macd, signal, hist):
            ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
            ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1)

            ax1.plot(prices)
            ax2.plot(macd, color='grey', linewidth=1.5, label='MACD')
            ax2.plot(signal, color='skyblue', linewidth=1.5, label='SIGNAL')

            for i in range(len(prices)):
                if str(hist[i])[0] == '-':
                    ax2.bar(prices.index[i], hist[i], color='#ef5350')
                else:
                    ax2.bar(prices.index[i], hist[i], color='#26a69a')

            plt.legend(loc='lower right')


        # plot_macd(api_stock['close'], stock_macd['macd'], stock_macd['signal'], stock_macd['hist'])

        # In this function I take the stock prices and MACD data as parameters to calculate the trades
        def implement_macd_strategy(prices, data):
            # empty lists to append later on the values
            buy_price = []
            sell_price = []
            macd_signal = []
            signal = 0

            # inside the for loop I am passing certain conditions to append the empty lists in case they are satisfied
            # the append value is 1 for when the stock is bought and -1 for the stock is sold.
            for i in range(len(data)):
                if data['macd'][i] > data['signal'][i]:
                    if signal != 1:
                        buy_price.append(prices[i])
                        sell_price.append(np.nan)
                        signal = 1
                        macd_signal.append(signal)
                    else:
                        buy_price.append(np.nan)
                        sell_price.append(np.nan)
                        macd_signal.append(0)
                elif data['macd'][i] < data['signal'][i]:
                    if signal != -1:
                        buy_price.append(np.nan)
                        sell_price.append(prices[i])
                        signal = -1
                        macd_signal.append(signal)
                    else:
                        buy_price.append(np.nan)
                        sell_price.append(np.nan)
                        macd_signal.append(0)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    macd_signal.append(0)

            return buy_price, sell_price, macd_signal


        buy_price, sell_price, macd_signal = implement_macd_strategy(api_stockMACD['close'], stock_macd)


        # in this part I am actually plotting the trading signals in the graph to make it better visually for the user
        def trading_signals_plotMACD():
            ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
            ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1)

            ax1.plot(api_stockMACD['close'], color='skyblue', linewidth=2, label=f'{api_tickMACD}')
            ax1.plot(api_stockMACD.index, sell_price, marker='^', color='green', markersize=10, label='BUY SIGNAL',
                     linewidth=0)
            ax1.plot(api_stockMACD.index, buy_price, marker='v', color='r', markersize=10, label='SELL SIGNAL',
                     linewidth=0)
            ax1.legend()
            ax1.set_title(f'{api_tickMACD} MACD SIGNALS')
            ax2.plot(stock_macd['macd'], color='grey', linewidth=1.5, label='MACD')
            ax2.plot(stock_macd['signal'], color='skyblue', linewidth=1.5, label='SIGNAL')

            for i in range(len(stock_macd)):
                if str(stock_macd['hist'][i])[0] == '-':
                    ax2.bar(stock_macd.index[i], stock_macd['hist'][i], color='#ef5350')
                else:
                    ax2.bar(stock_macd.index[i], stock_macd['hist'][i], color='#26a69a')

            plt.legend(loc='lower right')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(plt)


        trading_signals_plotMACD()

        # I am now creating an actual position for the stock for when it buys and when it buys and when it sells
        # the list will include values of 1 and 0.
        position = []
        # this for loop is to generate actual positions
        for i in range(len(macd_signal)):
            if macd_signal[i] > 1:
                position.append(0)
            else:
                position.append(1)
        # this for loop is to iterate over the values of the signal list
        for i in range(len(api_stockMACD['close'])):
            if macd_signal[i] == 1:
                position[i] = 1
            elif macd_signal[i] == -1:
                position[i] = 0
            else:
                position[i] = position[i - 1]

        macd = stock_macd['macd']
        signal = stock_macd['signal']
        close_price = api_stockMACD['close']
        macd_signal = pd.DataFrame(macd_signal).rename(columns={0: 'macd_signal'}).set_index(api_stockMACD.index)
        position = pd.DataFrame(position).rename(columns={0: 'macd_position'}).set_index(api_stockMACD.index)

        frames = [close_price, macd, signal, macd_signal, position]
        strategy = pd.concat(frames, join='inner', axis=1)
        strategy

        # Finally the backtesting part I am calculating the returns of the chosen stock
        # using the diff function (numpy) I am passing a for loop to iterate over the values of the stock_returns to
        # calculate the returns from the MACD part

        stock_returns = pd.DataFrame(np.diff(api_stockMACD['close'])).rename(columns={0: 'returns'})
        macd_strategy_ret = []

        for i in range(len(stock_returns)):
            try:
                returns = stock_returns['returns'][i] * strategy['macd_position'][i]
                macd_strategy_ret.append(returns)
            except:
                pass

        macd_strategy_ret_df = pd.DataFrame(macd_strategy_ret).rename(columns={0: 'macd_returns'})
        investment_value = float(st.text_input("Enter amount of investment", value=10000))
        number_of_stocks = floor(investment_value / api_stockMACD['close'][0])
        macd_investment_ret = []

        for i in range(len(macd_strategy_ret_df['macd_returns'])):
            returns = number_of_stocks * macd_strategy_ret_df['macd_returns'][i]
            macd_investment_ret.append(returns)

        macd_investment_ret_df = pd.DataFrame(macd_investment_ret).rename(columns={0: 'investment_returns'})
        total_investment_ret = round(sum(macd_investment_ret_df['investment_returns']), 2)
        profit_percentage = floor((total_investment_ret / investment_value) * 100)

        st.write(f'Profit gained from the MACD strategy by investing ${investment_value} in {api_tickMACD}' ': {}'.format(total_investment_ret))
        st.write('Profit percentage of the MACD strategy : {}%'.format(profit_percentage))

        if total_investment_ret <= 0:
            st.write("The current available equity is $0")



    if strategy_option == "RSI":

        RSI_description = st.write(
            "Being an oscillator, the values of RSI bound between 0 to 100. The traditional way to evaluate a market "
            "state using the Relative Strength Index is that an RSI reading of 70 or above reveals a state of "
            "overbought, and similarly, an RSI reading of 30 or below represents the market is in the state of "
            "oversold. These overbought and oversold can also be tuned concerning which stock or asset you choose.")

        def get_historical_dataRSI(symbol, start_date=None):
            api_key = '4UTGJUQ9TWN4VJ0I'
            api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&outputsize=full'
            raw_df = requests.get(api_url).json()
            df = pd.DataFrame(raw_df[f'Time Series (Daily)']).T
            df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
                                    '5. adjusted close': 'adj close', '6. volume': 'volume'})
            for i in df.columns:
                df[i] = df[i].astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.iloc[::-1].drop(['7. dividend amount', '8. split coefficient'], axis=1)
            if start_date:
                df = df[df.index >= start_date]
            return df

        # returns the historical data in a data frame
        api_tickRSI = st.text_input("Enter a stock", value="GOOGL", max_chars=5)
        api_stockRSI = get_historical_dataRSI(api_tickRSI, start_date="2017-01-01")


        def get_rsi(close, lookback):
            ret = close.diff()
            up = []
            down = []
            for i in range(len(ret)):
                if ret[i] < 0:
                    up.append(0)
                    down.append(ret[i])
                else:
                    up.append(ret[i])
                    down.append(0)
            up_series = pd.Series(up)
            down_series = pd.Series(down).abs()
            up_ewm = up_series.ewm(com=lookback - 1, adjust=False).mean()
            down_ewm = down_series.ewm(com=lookback - 1, adjust=False).mean()
            rs = up_ewm / down_ewm
            rsi = 100 - (100 / (1 + rs))
            rsi_df = pd.DataFrame(rsi).rename(columns={0: 'rsi'}).set_index(close.index)
            rsi_df = rsi_df.dropna()
            return rsi_df[3:]


        api_stockRSI['rsi_14'] = get_rsi(api_stockRSI['close'], 14)
        api_stockRSI = api_stockRSI.dropna()
        # stock_RSI


        ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4, colspan=1)
        ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=4, colspan=1)
        ax1.plot(api_stockRSI['close'], linewidth=2.5)
        ax1.set_title('IBM CLOSE PRICE')
        ax2.axhline(30, linestyle='--', linewidth=1.5, color='grey')
        ax2.axhline(70, linestyle='--', linewidth=1.5, color='grey')
        ax2.set_title('IBM RELATIVE STRENGTH INDEX')
        # plt.show()




        def implement_rsi_strategy(prices, rsi):
            buy_price = []
            sell_price = []
            rsi_signal = []
            signal = 0

            for i in range(len(rsi)):
                if rsi[i - 1] > 30 and rsi[i] < 30:
                    if signal != 1:
                        buy_price.append(prices[i])
                        sell_price.append(np.nan)
                        signal = 1
                        rsi_signal.append(signal)
                    else:
                        buy_price.append(np.nan)
                        sell_price.append(np.nan)
                        rsi_signal.append(0)
                elif rsi[i - 1] < 70 and rsi[i] > 70:
                    if signal != -1:
                        buy_price.append(np.nan)
                        sell_price.append(prices[i])
                        signal = -1
                        rsi_signal.append(signal)
                    else:
                        buy_price.append(np.nan)
                        sell_price.append(np.nan)
                        rsi_signal.append(0)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    rsi_signal.append(0)

            return buy_price, sell_price, rsi_signal


        buy_price, sell_price, rsi_signal = implement_rsi_strategy(api_stockRSI['close'], api_stockRSI['rsi_14'])

        def trading_signals_plotRSI():
            ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4, colspan=1)
            ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=4, colspan=1)
            ax1.plot(api_stockRSI['close'], linewidth=2.5, color='skyblue', label=f'{api_tickRSI}')
            ax1.plot(api_stockRSI.index, buy_price, marker='^', markersize=10, color='green', label='BUY SIGNAL')
            ax1.plot(api_stockRSI.index, sell_price, marker='v', markersize=10, color='r', label='SELL SIGNAL')
            ax1.set_title(f'{api_tickRSI} RSI SIGNALS')
            ax2.plot(api_stockRSI['rsi_14'], color='orange', linewidth=2.5)
            ax2.axhline(30, linestyle='--', linewidth=1.5, color='grey')
            ax2.axhline(70, linestyle='--', linewidth=1.5, color='grey')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(plt)
            #plt.show()

        trading_signals_plotRSI()

        position = []
        for i in range(len(rsi_signal)):
            if rsi_signal[i] > 1:
                position.append(0)
            else:
                position.append(1)

        for i in range(len(api_stockRSI['close'])):
            if rsi_signal[i] == 1:
                position[i] = 1
            elif rsi_signal[i] == -1:
                position[i] = 0
            else:
                position[i] = position[i - 1]

        rsi = api_stockRSI['rsi_14']
        close_price = api_stockRSI['close']
        rsi_signal = pd.DataFrame(rsi_signal).rename(columns={0: 'rsi_signal'}).set_index(api_stockRSI.index)
        position = pd.DataFrame(position).rename(columns={0: 'rsi_position'}).set_index(api_stockRSI.index)

        frames = [close_price, rsi, rsi_signal, position]
        strategy = pd.concat(frames, join='inner', axis=1)
        strategy


        # backtesting part
        rsi_ret = pd.DataFrame(np.diff(api_stockRSI['close'])).rename(columns={0: 'returns'})
        rsi_strategy_ret = []

        for i in range(len(rsi_ret)):
            returns = rsi_ret['returns'][i] * strategy['rsi_position'][i]
            rsi_strategy_ret.append(returns)

        rsi_strategy_ret_df = pd.DataFrame(rsi_strategy_ret).rename(columns={0: 'rsi_returns'})
        investment_value = float(st.text_input("Enter amount of investment", value=10000))
        # investment_value = 100000
        number_of_stocks = (investment_value / api_stockRSI['close'][-1])
        rsi_investment_ret = []

        for i in range(len(rsi_strategy_ret_df['rsi_returns'])):
            returns = number_of_stocks * rsi_strategy_ret_df['rsi_returns'][i]
            rsi_investment_ret.append(returns)

        rsi_investment_ret_df = pd.DataFrame(rsi_investment_ret).rename(columns={0: 'investment_returns'})
        total_investment_ret2 = round(sum(rsi_investment_ret_df['investment_returns']), 2)
        # just_the_return = total_investment_ret2 - investment_value
        # profit_percentage2 = floor((just_the_return / investment_value) * 100)
        profit_percentage2 = floor((total_investment_ret2 / investment_value) * 100)

        st.write(f'Profit gained from the RSI strategy by investing ${investment_value} in {api_tickRSI}' ': {}'.format(
            total_investment_ret2),
                    attrs=['bold'])
        st.write('Profit percentage of the RSI strategy : {}%'.format(profit_percentage2), attrs=['bold'])

        if investment_value <= 0:
            st.write("The current available equity is $0")

# tweets feed section taken from stocktwits and displayed into the web page
if option == "Stock Tweets":

    # The below code has been adapted to satisfy my needs for the features of this project from
    # https://www.youtube.com/watch?v=0ESc1bh3eIg&t=265s input text to search for specific stock for tweets
    stock_ticker = st.sidebar.text_input("Enter Ticker", value='AMZN', max_chars=5)
    link = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{stock_ticker}.json")

    stocktwits = link.json()

    for message in stocktwits['messages']:
        # added attributes in container so that appear better
        container = st.container()
        with st.container():
            container.image(message['user']['avatar_url'])
            container.write(message['user']['username'])
            container.write(message['created_at'])
            container.write(message['body'])
            container.write("--------------------------------------------------------------------------------------")

# login feature that will enable the user to login into their account and use the application's user features
if option == "Login":
    st.subheader("Account Login Page")
    username = st.text_input("User name")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        create_usertable()
        result = login_user(username, password)
        if result:
            st.success("Successfully logged in as {}".format(username))
            # and extra if statement for when the superuser logs into the application
            if username == "admin":
                st.subheader("User Profiles")
                user_result = view_all_users()
                clean_db = pd.DataFrame(user_result, columns=["Username", "Password"])
                st.dataframe(clean_db)

        else:
            st.error("The username or password in incorrect")

# Sign-Up for new accounts including a superuser "admin" for testing and database storing
if option == "Sign-Up":
    st.subheader("Sign-Up to an account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    if st.button("Sign-Up"):
        create_usertable()
        add_userdata(new_user, new_password)
        st.success("You have successfully created an account")
        st.info("Please login into your new account in the login page")
