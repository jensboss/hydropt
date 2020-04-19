#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:59:04 2020

@author: Jens
"""


import numpy as np



class Turbine():
    def __init__(self, name, nu, flow_rates, upper_basin, lower_basin):
        self.name = name
        self.nu = nu
        self.flow_rates = flow_rates
        self.upper_basin = upper_basin
        self.lower_basin = lower_basin
        
    def actions(self):
        return [Action(self, flow_rate=flow_rate) for flow_rate in self.flow_rates]
    
    def head(self):
        return self.upper_basin.levels() - self.lower_basin.levels()
    
    def power(self, flow_rate):
        return self.nu*(1000*9.81)*self.head()*flow_rate
    
    def __repr__(self):
        return f"Turbine('{self.name}')"