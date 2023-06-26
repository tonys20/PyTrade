import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime

FMP_KEY = 'ec973af37ee5857e4a08d15e67807b65'

class FinancialInstrument(): #Super class
    """
    similar to vectorized Basic Class .. Remain independent at this point.
    """
    def __init__(self, ticker:str, start:str, end:str):
        self._ticker = ticker
        self.start = start
        self.end = end
        self.get_data()
        self.log_returns()

    def __repr__(self):
        return f"Financial Instrument-->(ticker ={self._ticker}, start={self.start},end={self.end})"

    def get_data(self):
        raw = yf.download(self._ticker, self.start, self.end, progress=False).Close.to_frame()
        raw.rename(columns = {'Close':'Price'}, inplace = True)
        self.data = raw

    def log_returns(self):
        self.data['log_returns'] = np.log(self.data.Price/ self.data.Price.shift(1))

    def plot_prices(self):
        self.data.Price.plot(figsize = (12, 8))
        plt.title(f"Price Chat: {self._ticker}", fontsize=5)

    def plot_returns(self, kind:str ='ts'):

        if kind=='ts':
            self.data.log_returns.plot(figsize=(12, 8))
            plt.title(f'Returns: {self._ticker}', fontsize=15)

        elif kind=='hist':
            self.data.log_returns.hist(figsize=(12,8), bins = int(np.sqrt(len(self.data))))
            plt.title(f'Returns: {self._ticker}', fontsize=15)

    def set_ticker(self, ticker:str=None):
        if ticker is not None:
            self._ticker = ticker
            self.get_data()
            self.log_returns()


class RiskReturn(FinancialInstrument): #Sub Class

    def __init__(self,ticker:str, start:str, end:str, freq:str = None):
        self.freq = freq
        #the super() function gives access to super class methods
        super().__init__(ticker, start, end)

    def __repr__(self):
        return f'RiskReturn->(ticker ={self._ticker}, start={self.start},end={self.end})'

    def mean_return(self):
        if self.freq is None:
            return self.data.log_returns.mean()
        else:
            resampled = self.data.Price.resample(self.freq).last()
            resampled_ret = np.log(resampled/resampled.shift(1))
            return resampled_ret.mean()

    def std_return(self):
        if self.freq is None:
            return self.data.log_returns.std()
        else:
            resampled = self.data.Price.resample(self.freq).last()
            resampled_std = np.log(resampled/resampled.shift(1))
            return resampled_ret.std()

    def annualized_perf(self):
        mean_return = round(self.data.log_returns.mean()*252,3)
        risk = round(self.data.log_returns.std()*np.sqrt(252),3)
        print(f' Instrument: {self._ticker} \n Expected Return: {mean_return}\n Expected Risk: {risk}')



class IterativeBase():

    """
    Base Class for iterative backtesting.\n
    
    Params:
    ############################################################
    symbol: str
            ticker symbol of security
    start and end: str
            start and end dates in format "yyyy-mm-dd"\n
    initial balance: int
            set initial cash balance available for backtest\n

    Attributes:
    #############################################################
    curr_balance: int
            tracker variable for current balance at a given time bar\n
    units : int
            tracker variable for units of securities in position\n
    trades: int
            cumulator varibale for total number of trades executed\n
    position: 0,1,-1
            initialized at neutral position used to track signals in child strategy classes\n
    historical: pd.DataFrame
            historical OHLCV dataset as specified\n
    realtime: pd.DataFrame
            real time stock price quote feed (single entry)\n


    ##############################################################\n
    DEV Notes, add order types support: Market, Limit, Stop, etc.\n
    then, add set_holdings and portfolio construction method for convenient computation.\n
    finally but most challenging: simulate order book dynamics, add transaction costs simulation etc.\n
    """
    def __init__(self, symbol:str, start:str, end:str, amount:str, fmp_key:str):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = amount
        self.curr_balance = amount
        self.fmp_key = fmp_key
        self.units = 0
        self.trades = 0
        self.position = 0
        self.historical = self.get_data(real_time = False)
        self.realtime = self.get_data(real_time=True)




    def get_data(self, real_time:bool = False, freq='1min')->pd.DataFrame:
        """
        get historical or live feed price quote of a security
        :param real_time: default to False, get real time live data feed when set to True
        :param freq: data resolution in ['1min', '5min', '15min', '30min', '1hour','4hour']
        :return: pd.DataFrame
        """


        supported_freq = ['1min', '5min', '15min', '30min', '1hour','4hour']
        #process data type requested
        if real_time ==True:
            insert1 = 'quote-short'
            insert2 = ''
        else:
            if freq ==None:
                raise Exception('Must pass a frequency if requesting historical data')
            elif freq not in supported_freq:
                raise Exception(f'frequency not supported, choose one of the following \n {str(supported_freq)}')
            else:
                insert1 = 'historical-price-full'
                insert2 =f'from={self.start}&to={self.end}&'

        base_url = f'https://financialmodelingprep.com/api/v3/{insert1}/{self.symbol}?{insert2}apikey={self.fmp_key}'

        response = requests.get(base_url)
        if response.status_code == 200:
            data = response.json()
            output = pd.DataFrame(data)
            if real_time == False:
                output = output.drop(columns=['symbol'])
                output = pd.json_normalize(output['historical']).set_index('date')
                output.index = pd.to_datetime(output.index)
                output['log_ret'] = np.log(output.close/output.close.shift(-1))
                output.sort_index(inplace=True)
            else:
                output['Timestamp'] = datetime.now()
            pass
        else:
            raise Exception(f'Failed to get data for {self.symbol}')

        return output

    #core iteration method
    def get_values(self, bar:int):
        """
        Extract DataFrame row at bar location.
        :param bar:
        :return: date: datetime, price:float, spread:float
        """
        date = str(self.historical.index[bar].date())
        price = round(self.historical.open.iloc[bar], 5)
        spread = round(self.historical.close.iloc[bar]-self.historical.open.iloc[bar], 5)
        return date, price, spread

    def print_current_balance(self, bar:int)->None:
        """
        print currnt balance to the console
        :param bar: bar location
        :return: None
        """
        date, price, spread = self.get_values(bar)
        print(f'{date} | Current Balance: {round(self.curr_balance, 3)}')

    def buy_instrument(self, bar:int, units:int = None, amount:float = None)->(float,int):
        """
        Buy order action handler that takes in bar location and one of [units, amount]
        :param bar: bar location index
        :param units: quantity of securities to buy
        :param amount: amount of capital utilized to buy the security
        :return: price and units for report generation
        """
        date, price, spread = self.get_values(bar)

        if amount is not None:
            units = int(amount/price)
        self.curr_balance -= units*price
        self.units += units
        self.trades +=1

        print(f'{date} | Buying {units} {self.symbol} for {round(price,5)}, Remaining Balance:{self.curr_balance}')
        return price, units

    def sell_instrument(self, bar:int, units:int = None, amount:float = None)->(float,int):
        """
        Sell order action handler that takes in bar location and one of [units, amount]
        :param bar: bar location index
        :param units: quantity of securities to sell
        :param amount: amount of capital utilized to sell the security
        :return: price and units for report generation
        """

        date, price, spread = self.get_values(bar)

        if amount is not None:
            units = int(amount/price)
        self.curr_balance += units*price
        self.units -= units
        self.trades += 1

        print(f'{date} | Selling {units} {self.symbol} for {round(price,5)}, Remaining Balance:{self.curr_balance}')
        return price, units

    def print_current_position_value(self, bar:int)->None:
        """
        print the market value of the current position holdings to the console
        :param bar: bar location index
        :return: None
        """
        date, price, spread = self.get_values(bar)
        cpv = self.units*price
        print(f'{date} | Current position value is {round(cpv, 3)}')

    def print_current_nav(self, bar:int)->None:
        """
        print current net asset value to the console [position holdings + remaining balance
        :param bar: bar location index
        :return: None
        """
        date, price, spread = self.get_values(bar)
        nav = self.curr_balance + self.units * price
        print(f'{date} | Current Net Asset Value is {round(nav, 3)}')

    def close_pos(self, bar:int)->(float, int, str):
        """
        Closes out any positions currently active.
        :param bar: bar location index
        :return: price: float, units: int, ordertype:str in ['BUY', 'SELL', 'IDLE']
        """
        if self.position == 1:
            ordertype = 'SELL'
        elif self.position == -1:
            ordertype = 'BUY'
        else:
            ordertype = 'IDLE'


        date, price, spread = self.get_values(bar)
        print(75*'-')
        print(f'{date} | +++Closing Current Position+++ ')
        trade_units = self.units
        self.curr_balance += self.units * price
        print(f'{date} | Closing Current Position of {self.symbol} for {price}')
        self.units = 0
        self.trades += 1
        perf = (self.curr_balance/self.initial_balance-1)*100
        print(f'{date} | net performance = {perf}%')
        print(75*'-')
        return price, trade_units, ordertype

class IterativeBacktest(IterativeBase):
    """
    Child class from the super class IterativeBase. Sample implementation of algorithm.
    """
    def go_long(self, bar, units = None, amount = None):
        """
        Defines action associated with taking a long position in reaction to signals
        :param bar: bar location index
        :param units: int args passed to action handler
        :param amount: int args passed to action handler
        :return:  price: float, units: int, ordertype:str in ['BUY', 'SELL', 'IDLE']
        """
        if self.position == -1:
            price, units = self.buy_instrument(bar, units = -self.units)
        if units:
            price, units = self.buy_instrument(bar, units = units)
        elif amount:
            if amount == 'all':
                amount = self.curr_balance
            price, units = self.buy_instrument(bar, amount=amount)
        return price, units, 'BUY'


    def go_short(self, bar, units = None, amount = None):
        """
         Defines action associated with taking a short position in reaction to signals
        :param bar: bar location index
        :param units: int args passed to action handler
        :param amount: int args passed to action handler
        :return:  price: float, units: int, ordertype:str in ['BUY', 'SELL', 'IDLE']
        """
        if self.position == 1:
            price, units = self.sell_instrument(bar, units = self.units)
        if units:
            price, units = self.sell_instrument(bar, units = units)
        elif amount:
            if amount == 'all':
                amount = self.curr_balance
            price, units = self.sell_instrument(bar, amount=amount)
        return price, untis, 'SELL'


    def test_sma_crossover(self, SMA_S, SMA_L, report =False):
        """
        Implements core trading logic of the algorithm
        :param SMA_S: Short window of the SMA strategy
        :param SMA_L: Long window of the SMA strategy
        :param report: reprot=True toggles DataFrame output for the strategy (default False)
        :return: pandas.DataFrame if reprot == True
        """
        stout = f'TESTING SMA STRATEGY WITH PARAMETERS: SMA_S = {SMA_S} and SMA_L = {SMA_L}'
        print(75*'-')
        print(stout)
        print(75*'-')

        #RESET

        self.position = 0
        self.trades = 0
        self.curr_balance = self.initial_balance
        self.data = self.get_data()
        self.data['SMA_S'] = self.data['close'].rolling(SMA_S).mean()
        self.data['SMA_L'] = self.data['close'].rolling(SMA_L).mean()
        report_df = pd.DataFrame(columns=['OrderType', 'OrderQuantity', 'OrderPrice', 'RemainingBalance'])

        #Implement Strategy logics
        for bar in range(len(self.data)-1):
            if self.data['SMA_S'].iloc[bar] > self.data['SMA_L'].iloc[bar]:
                if self.position in [0,-1]:
                    order_price, order_quantity, ordertype = self.go_long(bar, amount = 'all')
                    self.position = 1
                    report_df = report_df.append({'OrderType': ordertype, 'OrderQuantity': order_quantity, 'OrderPrice': order_price, 'RemainingBalance': self.curr_balance}, ignore_index=True)

            elif self.data['SMA_S'].iloc[bar] < self.data['SMA_L'].iloc[bar]:
                if self.position in [0 , 1]:
                    order_price, order_quantity, ordertype =self.go_short(bar, amount='all')
                    self.position = -1
                    report_df = report_df.append({'OrderType': ordertype, 'OrderQuantity': order_quantity, 'OrderPrice': order_price, 'RemainingBalance': self.curr_balance}, ignore_index=True)

        order_price, order_quantity, ordertype = self.close_pos(bar+1)
        report_df = report_df.append({'OrderType': ordertype, 'OrderQuantity': order_quantity, 'OrderPrice': order_price, 'RemainingBalance': self.curr_balance}, ignore_index=True)
        return report_df









#TEST REGION
"""
ticker = 'AAPL'
start = '2013-01-01'
end = '2023-01-25'

test = IterativeBase(symbol =ticker, start=start, end = end, amount=100000, fmp_key=FMP_KEY)
hist = test.historical
rt = test.realtime
test.buy_instrument(0,units = 2000)
#test.sell_instrument(1,units = 190)

test.print_current_position_value(200)
test.print_current_nav(2000)

test.close_pos(-1)
sma = IterativeBacktest(symbol =ticker, start=start, end = end, amount=100000, fmp_key=FMP_KEY)
sma.test_sma_crossover(3,5)
"""