#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:49:29 2020

@author: Jens
"""


from bs4 import BeautifulSoup
import requests
import pandas as pd


def update_or_insert(df_1, df_2):
    df_1.update(df_2)
    df_concat = pd.concat((df_1, df_2), sort=False)
    mask =  ~df_concat.index.duplicated(keep='first')
    return df_concat.loc[mask, :]


market_areas = ['CH', 'DE-LU', 'FR']
url_temp = ("https://www.epexspot.com/en/market-data?"
            "market_area={}&"
            "trading_date={}&"
            "delivery_date={}&"
            "underlying_year=&"
            "modality=Auction&"
            "sub_modality=DayAhead&"
            "product=60&"
            "data_mode=table&"
            "period=")

date_end = pd.Timestamp.today().floor('D')
date_start = date_end - pd.offsets.Day()

date_range = pd.date_range(date_start, date_end, freq='D')

spot_prices = pd.read_csv('spot_data.csv', sep=';', index_col=[0], parse_dates=True)


for market_area in market_areas:
    df = pd.DataFrame()
    for date in date_range:
        trading_date = date
        delivery_date = date + pd.offsets.Day()
        
        url_request = url_temp.format(market_area,
                              trading_date.strftime('%Y-%m-%d'),
                              delivery_date.strftime('%Y-%m-%d')
                              )
        
        response = requests.get(url_request)
        
        soup = BeautifulSoup(response.content, features="lxml")
        table = soup.find("table", attrs={"class": "table-01 table-length-1"})

        temp = pd.read_html(table.prettify())[0]
        temp = temp.iloc[:,0:4]
        
        index = pd.date_range(delivery_date, 
                              delivery_date + pd.offsets.Day() - pd.offsets.Hour(), 
                              freq='H')
        temp = temp.set_index(index)
        
        df = pd.concat((df, temp), sort=False)
    
    try:
        data_from_file = pd.read_csv(f'results_{ market_area }.csv', 
                                     sep=';', 
                                     index_col=[0],
                                     parse_dates=True)
    except FileNotFoundError:
        data_from_file = pd.DataFrame()
        
    df = update_or_insert(data_from_file, df)
    
    df_price = df[['Price  (€/MWh)',]]
    df_price.columns = ['Price ' + market_area]
    spot_prices = update_or_insert(spot_prices, df_price)
    
    df.to_csv(f'results_{ market_area }.csv', sep=';')
    
spot_prices.to_csv('spot_data.csv', sep=';')
        





