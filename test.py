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
        # plt.show()


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
    investment_value = 100000
    number_of_stocks = floor(investment_value / api_stockMACD['close'][0])
    macd_investment_ret = []

    for i in range(len(macd_strategy_ret_df['macd_returns'])):
        returns = number_of_stocks * macd_strategy_ret_df['macd_returns'][i]
        macd_investment_ret.append(returns)

    macd_investment_ret_df = pd.DataFrame(macd_investment_ret).rename(columns={0: 'investment_returns'})
    total_investment_ret = round(sum(macd_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret / investment_value) * 100)
    # st.write(f'Profit gained from the MACD strategy by investing $100k in' + {api_tickMACD} + ': ' + total_investment_ret,
    #          attrs=['bold'])
    # st.write(f'Profit percentage of the MACD strategy : ' + profit_percentage, attrs=['bold'])

    st.write(cl(f'Profit gained from the MACD strategy by investing $100k in {api_tickMACD}' ': {}'.format(
        total_investment_ret),
                attrs=['bold']))
    st.write(cl('Profit percentage of the MACD strategy : {}%'.format(profit_percentage), attrs=['bold']))

    if total_investment_ret == 0:
        st.write("The current available equity is $0")

