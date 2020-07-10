#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:13:07 2020

@author: Jens
"""

import numpy as np
import matplotlib.pyplot as plt

from dynprog.model import Basin, Outflow, Turbine, PowerPlant, ActionStanding, \
    ActionMin, ActionMax
from dynprog.scenarios import Scenario, ScenarioOptimizer, Underlyings


basins = [Basin(name='basin_1', 
                volume=81, 
                num_states=81, 
                init_volume=10, 
                levels=(2000, 2120)),
          Basin(name='basin_2', 
                volume=31, 
                num_states=41, 
                init_volume=10, 
                levels=(1200, 1250))
          ]

outflow = Outflow(outflow_level=600)

turbines = [Turbine('turbine_1', 
                    max_power = 33000000.0,
                    base_load = 10000000.0,
                    efficiency=0.8, 
                    upper_basin=basins[0], 
                    lower_basin=basins[1],
                    actions=[ActionStanding(), ActionMin(), ActionMax()]),
            Turbine('turbine_2', 
                    max_power = 15000000.0,
                    base_load =  7000000.0,
                    efficiency=0.8,
                    upper_basin=basins[1], 
                    lower_basin=outflow,
                    actions=[ActionStanding(), ActionMin(), ActionMax()])
            ]

power_plant = PowerPlant(basins, turbines)    


def date_range(start_time, end_time, sampling_time=None):
    if sampling_time is None:
        start_time = np.datetime64(start_time)
        end_time = np.datetime64(end_time)
    else:
        start_time = np.datetime64(start_time, sampling_time)
        end_time = np.datetime64(end_time, sampling_time)
        
    return np.arange(start_time, end_time)
    

start_time = '2020-04-01T00' 
end_time =  '2020-04-14'
time = date_range(start_time, end_time)

n_steps = len(time)
hpfc = 10*(np.sin(2*np.pi*2*np.arange(n_steps)/n_steps) + 1)
inflow = 0.8*np.ones((n_steps,2))

underlyings = Underlyings(time, hpfc, inflow)
scenario = Scenario(power_plant, underlyings, name='base')

scenario.power_plant


optimizer = ScenarioOptimizer(scenario)
optimizer.run()

plt.figure(2)
plt.clf()
plt.plot(hpfc, marker='.', label='hpfc')
plt.plot(10*inflow, marker='.', label='inflow')
plt.plot(optimizer.turbine_actions/1e6, marker='.', label='action')
plt.plot(np.arange(n_steps+1)-1,optimizer.volume, marker='.', label='vol')
plt.legend()
plt.show()

# print('EURO', np.sum(np.sum(optimizer.turbine_actions, axis=1)*hpfc)/1e6)
