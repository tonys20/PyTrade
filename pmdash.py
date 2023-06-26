import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tony2 import iterative


symbol = 'AAPL'
start = '2023-01-01'
end = '2023-05-01'
stock = iterative.IterativeBacktest(symbol = symbol, start = start, end = end, amount =1000000, fmp_key=iterative.FMP_KEY)

