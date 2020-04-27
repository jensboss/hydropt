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


ActionMax()basins = [Basin(name='basin_1', 
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

plant = PowerPlant(basins, turbines)    

n_steps = 24*7*2

hpfc = 10*(np.sin(2*np.pi*2*np.arange(n_steps)/n_steps) + 1)

inflow = 0.8*np.ones((n_steps,2))

underlyings = Underlyings(hpfc, inflow, 3600)
scenario = Scenario(plant, underlyings, name='base')

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

print('EURO', np.sum(np.sum(optimizer.turbine_actions, axis=1)*hpfc)/1e6)
