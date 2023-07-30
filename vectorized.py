#Primary optimization engine
import scipy.optimize as opt
import scipy.stats

#Data manipulation and wrangling
import pandas as pd
import numpy as np
from datetime import datetime as dt

#Data API access
import yfinance as yf
from full_fred.fred import Fred
import requests as re

#Fun segway: Email and WeChat message bots
import smtplib
from email.message import EmailMessage
from datetime import datetime
from datetime import timedelta
import itchat

#Plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

#%%

# Some frequently used codes

# Define acceptable frequencies
_FREQ ={'1mo':'Monthly',
        '1d':'Daily',
        '1m': 'Minute'}

_TYPE = {'EQT':'Equity',
         'MACRO': 'Macroeconomic Indicator',
         'OPT': 'Options',
         'FUT': 'Futures',
         'ETF': 'ETF'
         }

#set helper variable for today's datetime string
TODAY=datetime.today()
TODAY_STR =TODAY.strftime('%Y-%m-%d')
FMP_BASE = 'https://financialmodelingprep.com/api'

#API keys - Confidential-> Remove before publishing
fmp_key = 'ec973af37ee5857e4a08d15e67807b65'
fred_key = 'c9731f8305946ceb1acea22613ea3c31'


class Basic():
    """
    The basic class serves as a foundation for the entire module.

    Attributes:
    ###########################################################################
    ticker: ticker symbol for equity, futures and options or series ID for macroeconomic data series
    cat: category of series. Refer to category dictionary for definition
    fred_key fmp_key: initialize the class with API keys for convenience
    """

# initialize the object with ticker, asset type and an optional argument FRED KEY
    def __init__(self, ticker:str, cat:str, fred_key =fred_key, fmp_key=fmp_key)->None:
        """
        Initializes the module base class Basic
        :param ticker: str ticker symbol or series ID of requested data
        :param cat: category of securities requested. Refer to vectorize._TYPE for details
        :param fred_key: FRED API Key. Must provide if requesting FRED Macro data.
        :param fmp_key: FMP API Key. Must provide if requesting Security specific data.
        """
        self.ticker = ticker
        self.cat = cat
        #check if type of asset is in recognized type list
        if self.cat not in _TYPE.keys():
            raise ValueError('Asset Type not Recognized!')

        elif self.cat =='MACRO':
            self.fred_key = fred_key
            self.fred = Fred(fred_key)
            if self.fred_key == None:
                raise Exception('Must give a FRED API key if using MACRO data')

        elif self.cat == 'EQT':
            self.fmp_key=fmp_key
            if self.fmp_key == None:
                raise Exception('Must give an FMP API key if using non macro data')


    #The print func should print company registration info for stocks and brief description for economic data series
    def __str__(self):
        query = 'company - core - information'
        url =f'{FMP_BASE}/v4/{query}?symbol={self.ticker}&apikey={self.fmp_key}'
        response = re.get(url)
        info = response.json()
        entry_ls = info[0].keys()
        stout=str()
        for name in entry_ls:
            stout += name
            stout += ':  '
            stout += info[0][name]
            stout += ' \n'
        return stout

    def __eq__(self, other):
        if self.ticker == other.ticker and self.cat == other.cat:
            return True
        else:
            return False


    #method to get data from either FRED website or yahoo finance
    def get_data(self,  date_start:str, date_end:str, freq:str ='1d') -> pd.DataFrame :
        """
        Data retrieval method
        :param date_start: beginning date of data series
        :param date_end: end date of data series
        :param freq: 1m 1d 1mo etc.. refer to yahoo finance for this parameter
        :return: pandas.DatFrame
        """
        #parse date strings
        dt_0 = dt.strptime(date_start, '%Y-%m-%d')
        dt_1 = dt.strptime(date_end, '%Y-%m-%d')
        #Compare to check date order
        if dt_1 < dt_0:
            raise ValueError('date range invalid!')
        if self.cat not in _TYPE.keys():
            raise ValueError('Unsupported Asset Type!')

        output = pd.DataFrame()
        if self.cat != 'MACRO':
            #accumulate yahoo finance series into dataframe
            source = yf.download(tickers = self.ticker, start =date_start, end = date_end, interval = freq, progress = False)
            col_names = list(source.columns)
            for name in col_names:
                if 'Adj' in name:
                    adj_ret_colname = name
                output[f'{self.ticker} {name}'] = source[name]

            #calculate continuously compounded returns and append it to dataframe
            output[f'{self.ticker} {adj_ret_colname} Return'] = output[f'{self.ticker} {adj_ret_colname}'] /  output[f'{self.ticker} {adj_ret_colname}'].shift(1) -1
        elif self.cat in ['EQT', 'OPT', 'FUT']:
            #define function to get data from FRED database
            fred_sereis = lambda ticker: self.fred.get_series_df(ticker,
                                                           observation_start=date_start,
                                                           observation_end=date_end,
                                                           ).drop(columns = ['realtime_start', 'realtime_end'])\
                .set_index('date', inplace=False)

            output[f'{self.ticker} Value'] = fred_sereis(self.ticker)

        return output


    def get_ratios(self, stout:bool = True)->pd.DataFrame:
        """
        FMP API wrapper for retrieving financial ratios dataset of a specified security
        :param stout: bool- toggle to print ratio to console
        :return: pandas.DataFrame of financial ratios
        """
        if self.cat != 'EQT':
            raise Exception('Only single company PE supported for now')

        url=f'{FMP_BASE}/v3/ratios-ttm/{self.ticker}?apikey={self.fmp_key}'
        response = re.get(url)
        if response.status_code == 200:
            ratios = response.json()

        else:
            raise Exception('Failed to download data')

        output = pd.DataFrame(ratios)
        return output


    def get_news(self, stout:bool = False, limit:int =50)->str:
        """
        Retrieves a list of all latest news about a specified security (only supports equity data)
        :param stout: bool- toggle to print ratio to console
        :param limit: maximum number of news to retrieve
        :return: body: news text content, news:json file from the request
        """
        #api url
        url = f'{FMP_BASE}/api/v3/stock_news?tickers={self.ticker}&limit={limit}&apikey={self.fmp_key}'
        response = re.get(url)
        # check status code
        if response.status_code == 200:
            news = response.json()
        else:
            raise Exception(f'Failed to retrieve news for {self.ticker}')
        #console output
        if stout==True:
            for item in news:
                print(f"Title: {item['title']}\nURL: {item['url']}\nPublished at: {item['publishedDate']}\n")
            return news
        else:
            body = str()
            for item in news:
                body+=(f"Title: {item['title']}\nURL: {item['url']}\nPublished at: {item['publishedDate']}\n\n\n")
            return body, news

    #This method pulls the earnings calendar item from the fmp API
    def get_earnings_calendar(self, date_start, date_end, type='confirmed', stout=False)->str:

        """
        :param date_start: starting date of earnings calendar
        :param date_end: end date of earnings calendar
        :param type: confirmed or unconfirmed --- UPDATE THIS TO A BOOLEAN VARIABLE ITS HARD TO USE AS FUCK..
        :param stout: bool- toggle to print ratio to console
        :return: content:str of calendar
        """

        if type == 'confirmed':
            url = f'{FMP_BASE}/v4/earning-calendar-confirmed?from={date_start}&to={date_end}&apikey={self.fmp_key}'

        elif type =='unconfirmed':
            url = f'{FMP_BASE}/api/v3/earning_calendar?apikey={self.fmp_key}'
        else:
            raise Exception('Calendar type unknown!')

        response = re.get(url)
        if response.status_code == 200:
            calendar = response.json()

        else:
            raise Exception(f'{response.status_code}: Failed to retrieve the confirmed calendar')

        col_names= [name for name in calendar[0].keys()]

        content = str()
        for item in calendar:
            print(item)
            for name in col_names:
               content += f' {name}:  {item[name]} \n'

        if stout == True:
            print(content)
        return content


    def push_news(self,api_key:str, to_email:str, type='news')->None:
        """
        method to push news email udpate to specified email address. dependent on get_news method.

        :param api_key: FMP API Key included in package
        :param to_email: recipient address
        :param type: content type in one of ['news', 'earnings']
        :return: None
        """
        from_email = 'automatedtony23@gmx.com'
        smtp_server = 'smtp.gmx.us'
        smtp_port = 465
        smtp_user = from_email
        smtp_password = 'St201212+'
        msg = EmailMessage()
        if type == 'news':
            body = str(self.get_news(api_key=api_key, stout=False)[0])
            subject = f'{self.ticker} News Push from Tony'
        elif type == 'earnings':
            body = str(self.get_earnings_calendar(api_key, date_start=TODAY_STR, date_end=(TODAY+timedelta(days=7)).strftime('%Y-%m-%d'),type='confirmed'))
            subject = f'Earnings Calendar for next week'
        else:
            raise Exception('Bad Push Type Requested')

        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email

        try:
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            print("Email sent successfully!")
        except Exception as e:
            print(f"Error sending email: {e}")

    #WIP Wechat msg bot
    def push_wechat(self, api_key:str):
        """
        functionality under development
        :param api_key: FMP API Key included in package
        :return: None
        """
        itchat.auto_login(hotReload=True)
        nickname = 'Tony'
        users = itchat.search_friends(nickname=nickname)
        if users:
            user = users[0]
            user_name = user['UserName']
            message = str(self.get_news(api_key=api_key, stout=False)[0])
            itchat.send(message,toUserName=user_name)
        else:
            print(f'failed to find friend: {nickname}!')
        itchat.logout()

#%%
# The second class implements some basic endogenous trading strategies
STRAT_TYPE = ['endo','exo']


class Strategy():
    """
    The Strategy class implements the strategies I've accumulated so far. Goal is to classify strategies based on characteristics
    endogenous versus endo and exo combined.

    Attributes:
    ===========
    ticker: similar to the base class ticker _PROTECTED ATTRIBUTE\n
    stock: instantiate base class object \n
    data: base class method get data output\n
    self.strat_params: currently not in use, will include stop loss, limit gain or other stoppage conditions.\n
    self.starting_cash: set starting cash position for the strategy.\n
    self._date_start self._date_end: _PROTECTED ATTRIBUTE for date range of strategy.\n
    self._freq: 1d 1m 1mo ... freq str similar to base class.\n

    Methods:
    ===========
    REMOVED - set_ticker: change the _PROTECTED ATTRIBUTE self._ticker and update dependent attributes.\n

    """

    def __init__(self, ticker:str, date_start:str, date_end:str, freq:str='1d', starting_cash:float = 10000, strat_params = None):
        """
        Initializes
        :param ticker:
        :param date_start:
        :param date_end:
        :param freq:
        :param starting_cash:
        :param strat_params:
        """
        self.ticker = ticker
        self.strat_params = strat_params
        self.date_start = date_start
        self.date_end = date_end
        self.freq = freq
        self.starting_cash = starting_cash
        self.stock = Basic(ticker = self.ticker, cat = 'EQT', fmp_key=fmp_key)
        self.data = self.stock.get_data(date_start, date_end, freq=freq)



    def __repr__(self):
        return f'Strategy object on {self.ticker}'


    #simple moving average crossover signals
    def sma_crossover(self, short_window:int, long_window:int)->pd.DataFrame:
        """
        Sample implementation of simple SMA crossover strategy.

        :param short_window: short window length parameter of the strategy
        :param long_window: long window length parameter of the strategy
        :return: dataframe with performance metrics, positions and signals
        """
        data = self.data.copy()
        data[f'{self.freq}_return'] = data[f'{self.ticker} Adj Close'].pct_change()
        data[f'sma_{short_window}'] = data[f'{self.ticker} Adj Close'].rolling(short_window).mean()
        data[f'sma_{long_window}'] = data[f'{self.ticker} Adj Close'].rolling(long_window).mean()
        data['sma_gap'] = data[f'sma_{short_window}'] - data[f'sma_{long_window}']
        data['sma_buy_signal'] = np.where((data['sma_gap']>0) & (data['sma_gap'].shift(1)<0),1,0)
        data['sma_sell_signal'] = np.where((data['sma_gap']<0) & (data['sma_gap'].shift(1)>0),-1,0)
        data['sma_signal'] = data['sma_buy_signal'] + data['sma_sell_signal']
        data['sma_cum_return'] = (data['sma_signal'].shift(-1)*data[f'{self.freq}_return']+1).cumprod()-1

        return data
    #more strategies under development





    #Performance measurement metrics later to be put into a separate class.


    def compute_sr(df:pd.DataFrame,rf:float=0, nametag ='Close Return')->float:
        """
        Caculates conventional sharpe ratio given a portfolio performance dataframe.

        :param rf: risk free rate
        :param nametag: column name in the dataframe that contains returns metric for calculating sharpe ratio
        :return: sharpe_ratio, nametag
        """
        df = df.dropna()
        name = None

        for colname in df.columns:
            if nametag in colname:
                name = colname
                mean_return = df[colname].mean()
                std = df[colname].std()

        sharpe_ratio = (mean_return - rf) / std
        return sharpe_ratio, name

    #Calculate probabilitistic sharpe ratio. NOTE: DEPENDENT ON compute_sr method
    def compute_psr(df:pd.DataFrame, benchmark:float = 0, nametag='Close Return')->float:
        """
        computes probabilistic sharpe ratio. Dependent on sharpe ratio calculation.

        :param benchmark: bechmark sharpe ratio for deviation calcualtion
        :param nametag: column name for returns
        :return: psr in float
        """
        df = df.dropna()
        sr, colname = Strategy.compute_sr(df, nametag=nametag)
        skew = scipy.stats.skew(df[colname])
        kurtosis = scipy.stats.kurtosis(df[colname], fisher=True)
        n = len(df)
        sigma_sr = ((1/(n-1))*(1+0.5*sr**2+skew*sr+(kurtosis/4)*sr**2)) ** 0.5

        ratio = (sr-benchmark)/sigma_sr
        psr = scipy.stats.norm.cdf(ratio)
        return psr







#%%
# test code

####### TEST CODE FOR DEV USE ONLY ###########
"""
ticker = 'AAPL'
cat = 'EQT'

stock = Basic(ticker = ticker, cat = cat, fmp_key=fmp_key)
date_start = '2023-4-01'
date_end = '2023-5-02'
df = stock.get_data(date_start,date_end,freq='1d')
ratios = stock.get_ratios(stout = True)

test_strat = Strategy(ticker = ticker, date_start=date_start, date_end=date_end, freq='1d')
sma  = test_strat.sma_crossover(5 , 10)

psr = Strategy.compute_psr(df =sma, nametag='sma_cum_return')
"""