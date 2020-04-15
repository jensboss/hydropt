#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:06:02 2020

@author: Jens
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from model import Basin, Outflow, Turbine, PlantModel
from dyn_prog import backward_induction, forward_propagation




if __name__=='__main__':
    basins = [Basin(name='basin_1', vol=100, num_states=81, content=10, height=2000),
              Basin(name='basin_2', vol=30, num_states=41, content=10, height=1200)]
    
    outflow = Outflow(height=600)
    
    turbines = [Turbine('turbine_1', nu=0.8, flow_rates=[0,5], 
                        upper_basin=basins[0], lower_basin=basins[1]),
                Turbine('turbine_2a', nu=0.8, flow_rates=[0,2], 
                        upper_basin=basins[1], lower_basin=outflow),
                Turbine('turbine_2b', nu=0.7, flow_rates=[0,2], 
                        upper_basin=basins[1], lower_basin=outflow)]
    
    pp = PlantModel(basins, turbines)
    
    pp_actions = pp.actions()
    print(pp_actions)
    print(pp_actions.turbine_prod())
    print(pp_actions.basin_flow_rates())
       
    actions = np.array(pp_actions.turbine_prod())
    net_actions = np.array(pp_actions.basin_flow_rates())
    
    n_steps = 24*7*1
    
    hpfc = 10*(np.sin(2*np.pi*2*np.arange(n_steps)/n_steps) + 1)
    inflow = 0.8*np.ones((n_steps,2))
    penalty = 10000000
    water_value_end = 0
    
    volume = pp.basin_volumes()
    num_states = pp.basin_num_states()
    basins_contents = pp.basin_contents()
    

    t_start = time.time()
    action_grid, value_grid = backward_induction(n_steps, volume, num_states, 
                                                  actions, net_actions, inflow, 
                                                  hpfc, water_value_end, penalty)
    t_end = time.time()
    print(t_end-t_start)
    
    
    t_start = time.time()
    actions_taken, net_actions_taken, vol = forward_propagation(n_steps, volume, 
                                                                num_states, basins_contents,
                                                                actions, 
                                                                net_actions,inflow, 
                                                                action_grid)
    
    t_end = time.time()
    print(t_end-t_start)
    

    plt.figure(1)
    plt.clf()
    plt.plot(hpfc, marker='.', label='hpfc')
    plt.plot(10*inflow, marker='.', label='inflow')
    plt.plot(actions_taken/1000000, marker='.', label='action')
    plt.plot(np.arange(n_steps+1)-1,vol, marker='.', label='vol')
    plt.legend()
    plt.show()