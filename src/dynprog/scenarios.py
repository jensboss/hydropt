#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:06:02 2020

@author: Jens
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from model import Basin, Outflow, Turbine, PlantModel, BasinLevels
from dyn_prog import backward_induction, forward_propagation


class Underlyings():
    def __init__(self, price_curve=None, inflow=None, sampling_time=1):
        self.price_curve = price_curve
        self.inflow = inflow
        self.sampling_time = sampling_time
        
    def n_steps(self):
        return self.price_curve.shape[0]
        
class Scenario():
    def __init__(self, model, underlyings, water_value_end=0, name=None):
        self.model = model
        self.underlyings = underlyings
        self.water_value_end = water_value_end
        self.name = name
        
        
class ScenarioOptimizer():
    def __init__(self, scenario=None, basin_limit_penalty=100000000000):
        self.scenario = scenario
        self.basin_limit_penalty = basin_limit_penalty
        self.action_grid = None
        self.value_grid = None
        self.turbine_actions = None
        self.basin_actions = None
        self.volume = None
        
    def run(self):
        n_steps = self.scenario.underlyings.n_steps()
        price_curve = self.scenario.underlyings.price_curve
        inflow = self.scenario.underlyings.inflow
        
        model_actions = self.scenario.model.actions()
        turbine_actions = np.array(model_actions.turbine_power())
        basin_actions = np.array(model_actions.basin_flow_rates())
        
        volume = self.scenario.model.basin_volumes()
        num_states = self.scenario.model.basin_num_states()
        basins_contents = self.scenario.model.basin_contents()
        
        penalty = self.basin_limit_penalty  
        
        water_value_end = self.scenario.water_value_end
        
        t_start = time.time()
        action_grid, value_grid = backward_induction(n_steps, volume, num_states, 
                                                      turbine_actions, basin_actions, inflow, 
                                                      price_curve, water_value_end, penalty)
        t_end = time.time()
        print(t_end-t_start)
        
        t_start = time.time()
        turbine_act_taken, basin_act_taken, vol = forward_propagation(n_steps, volume, 
                                                                    num_states, basins_contents,
                                                                    turbine_actions, 
                                                                    basin_actions, inflow, 
                                                                    action_grid)
        
        t_end = time.time()
        print(t_end-t_start)
        
        self.action_grid = action_grid
        self.value_grid = value_grid
        
        self.turbine_actions = turbine_act_taken
        self.basin_actions = basin_act_taken
        self.volume = vol
        
            

if __name__=='__main__':
    basins = [Basin(name='basin_1', vol=101, num_states=81, content=10, height=BasinLevels(2000,2120)),
              Basin(name='basin_2', vol=31, num_states=41, content=10, height=BasinLevels(1200,1250))]
    
    outflow = Outflow(height=600)
    
    turbines = [Turbine('turbine_1', nu=0.8, flow_rates=[0, 5], 
                        upper_basin=basins[0], lower_basin=basins[1]),
                Turbine('turbine_2a', nu=0.8, flow_rates=[0, 2], 
                        upper_basin=basins[1], lower_basin=outflow),
                Turbine('turbine_2b', nu=0.7, flow_rates=[0, 2], 
                        upper_basin=basins[1], lower_basin=outflow)]
    
    plant = PlantModel(basins, turbines)    
    
    n_steps = 24*7*4
    hpfc = 10*(np.sin(2*np.pi*2*np.arange(n_steps)/n_steps) + 1)
    inflow = 0.8*np.ones((n_steps,2))
    
    underlyings = Underlyings(hpfc, inflow, 3600)
    scenario = Scenario(plant, underlyings, name='base')
    
    optimizer = ScenarioOptimizer(scenario)
    optimizer.run()
    
    plt.figure(1)
    plt.clf()
    plt.plot(hpfc, marker='.', label='hpfc')
    plt.plot(10*inflow, marker='.', label='inflow')
    plt.plot(optimizer.turbine_actions/1000000, marker='.', label='action')
    plt.plot(np.arange(n_steps+1)-1,optimizer.volume, marker='.', label='vol')
    plt.legend()
    plt.show()