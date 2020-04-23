#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:13:07 2020

@author: Jens
"""

import numpy as np
import matplotlib.pyplot as plt

from dynprog.model import Basin, BasinLevels, Outflow, Turbine, Plant
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
                    nu=0.8, 
                    flow_rates=[0, 5], 
                    upper_basin=basins[0], 
                    lower_basin=basins[1]),
            Turbine('turbine_2', 
                    nu=0.8, 
                    flow_rates=[0, 2], 
                    upper_basin=basins[1], 
                    lower_basin=outflow)
            ]

plant = Plant(basins, turbines)    

n_steps = 24*7*1
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
plt.plot(optimizer.turbine_actions/1000000, marker='.', label='action')
plt.plot(np.arange(n_steps+1)-1,optimizer.volume, marker='.', label='vol')
plt.legend()
plt.show()