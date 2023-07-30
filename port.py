#IMPORT REGION
from tony2 import iterative
from tony2 import vectorized
import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache
#######################################################################


class Portfolio:
    """
    portfolio base class
    :param: start, end: str defines holding period
    :param: port=None: dict{symbol: quantity}
    """

    def __init__(self, start:str, end:str, port:dict=None, fmp_key:str=iterative.FMP_KEY):
        self.fmp_key = fmp_key

        self.start = start
        self.end = end
        self.start_dt = datetime.strptime(self.start, '%Y-%m-%d')
        self.end_dt = datetime.strptime(self.end, '%Y-%m-%d')
        self.hold_years = (self.end_dt - self.start_dt).days/365

        self._instruments = port or {}
        self.datasets = self._load_securities()

    @lru_cache(maxsize=None)
    def _load_securities(self):
        """
        :return: dict{symbol: pd.DataFrame}
        """
        securities = [
            iterative.IterativeBase(symbol=symbol, start=self.start, end=self.end, amount=self._instruments[symbol], fmp_key=self.fmp_key)
            for symbol in self._instruments.keys()
        ]
        datasets = {security.symbol: security.get_data(freq='1min') for security in securities}
        return datasets

    @classmethod
    def from_symbols(cls, symbols:list, start:str, end:str, fmp_key:str=iterative.FMP_KEY, amount:float=100):
        """
        class method to construct portfolio dictionary supported by initializer

        :param symbols: dict{symbol: quantity}
        :param start: str '%Y-%m-%d'
        :param end: str '%Y-%m-%d'
        :param fmp_key: str: included in package
        :param amount: idle arg for security initialization
        :return: dict{symbol, capitalization}
        """
        securities = [
            iterative.IterativeBase(symbol=symbol, start=start, end=end, amount=symbols[symbol], fmp_key=fmp_key)
            for symbol in symbols.keys()
        ]

        port = {
            security.symbol: security.get_data(freq='1min').close.iloc[-1] * symbols[security.symbol]
            for security in securities
        }

        port_capitalization = sum(port.values())

        for symbol in port.keys():
            port[symbol] /= port_capitalization

        return port

    def update_port(self, symbol:str, weight:float):

        """
        Action handler to update weights of securities within portfolio
        :param symbol: symbol of security to update
        :param weight: new updated weight
        :return: None
        """

        if weight<0 or weitght>1:
            raise ValueError('Unaccepted weight range')
        if symbol not in self._instruments.keys():
            raise KeyError('udpate_port is only applied to existing securities in the portfolio. Use add_instrument instead')

        remaining_weight = 1-weight
        total_other_weights = sum(weight for sym, weight in self._instruments.items() if sym != symbol)
        for sym, weight in self._instruments.weights:
            if sym != symbol:
                self._instruments[sym] =(weight/total_other_weights)/remaining_weight

        self._instruments[symbol] = weight
        #Reload securities
        self._load_securities()

    def performance_metrics(self)->pd.DataFrame:
        """
        Basic performance metrics calculation
        :return: characteristics: pd.DataFrame of historical metrics of portfolio, expected: weighted expected value of volatility, return and sharpe
        """
        characteristics = pd.DataFrame(columns = self._instruments.keys())
        expected = pd.DataFrame(columns = ['value'])

        for symbol in self._instruments.keys():
            characteristics.loc['max close',symbol] = self.datasets[symbol].close.max()
            characteristics.loc['min close',symbol] = self.datasets[symbol].close.min()
            characteristics.loc['mean return',symbol] = self.datasets[symbol].log_ret.mean()
            characteristics.loc['standard dev', symbol]= self.datasets[symbol].log_ret.std()
            for other_symbol in self._instruments.keys():
                characteristics.loc[other_symbol, symbol] = np.corrcoef((self.datasets[symbol].log_ret).dropna(), (self.datasets[other_symbol].log_ret).dropna())[0][1]

        weights_array = np.array(list(self._instruments.values()))
        std_dev_array = np.array(characteristics.loc['standard dev'])
        variances = np.outer(std_dev_array, std_dev_array)
        corr_matrix = characteristics.loc[list(self._instruments.keys()),list(self._instruments.keys())].values
        cov_matrix = variances * corr_matrix

        portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
        portfolio_std_dev = np.sqrt(portfolio_variance)
        expected.loc['Port Annualized Returns', 'value'] = 252 * sum([characteristics.loc['mean return', symbol] * weight
                                                                      for symbol, weight in self._instruments.items()])
        expected.loc['Port Annualized Volatility', 'value'] = np.sqrt(252) * portfolio_std_dev
        expected.loc['Sharpe Ratio'] = (expected.loc['Port Annualized Returns', 'value'] -0.015)/expected.loc['Port Annualized Volatility', 'value']

        return characteristics, expected


#TEST REGION

"""
symbols = {'AAPL': 1000, 'GE': 1000, 'MMM': 1000, 'MSFT':1000, 'EMR':1000, 'IFF':1000}

start = '2013-01-01'
end = '2023-01-01'
fmp_key = iterative.FMP_KEY
amount = 10000

assets = Portfolio.from_symbols(symbols=symbols, start=start, end =end)
port_1 = Portfolio(start=start, end =end, port=assets)
df1, df2 = port_1.performance_metrics()
df2
"""