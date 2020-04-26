#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:06:02 2020

@author: Jens
"""

import numpy as np
import time

from dynprog.core import backward_induction, forward_propagation


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
    def __init__(self, scenario=None, basin_limit_penalty=1e14):
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
        basins_init_volumes = self.scenario.model.basin_init_volumes()
        
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
                                                                    num_states, basins_init_volumes,
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
        
            

