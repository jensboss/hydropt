from hydropt.model import Basin, Outflow, Turbine, PowerPlant
from hydropt.action import Standing, MinPower, MaxPower
from hydropt.scenarios import Scenario, Underlyings

import importlib.resources as pkg_resources
import pandas as pd

def load_spot_data():
    ctx = pkg_resources.path('hydropt', 'data')

    with ctx as path:
        data = pd.read_csv(
            path / 'spot_prices_2019.csv', 
            sep=';', 
            index_col=0,
            parse_dates=True)
        
    return data