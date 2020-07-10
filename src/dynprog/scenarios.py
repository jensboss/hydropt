#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:06:02 2020

@author: Jens
"""

import numpy as np
import time

from dynprog.core import backward_induction, forward_propagation
from dynprog.constraints import ConstrainedIntervals


class Underlyings():
    def __init__(self, time, price_curve=None, inflow=None):
        self.time = time
        self.price_curve = price_curve
        self.inflow = inflow
        
    def n_steps(self):
        return self.time.shape[0]
    
        
class Scenario():
    def __init__(self, power_plant, underlyings, constraints=None, water_value_end=0, name=None):
        self.power_plant = power_plant
        self.underlyings = underlyings
        
        if constraints is None:
            self.constraints = ConstrainedIntervals()
        else:
            self.constraints = constraints
            
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
        
        power_plant_actions = self.scenario.power_plant.actions()
        
        turbine_actions = np.array(power_plant_actions.turbine_power())
        basin_actions = np.array(power_plant_actions.basin_flow_rates())
        
        volume = self.scenario.power_plant.basin_volumes()
        num_states = self.scenario.power_plant.basin_num_states()
        basins_init_volumes = self.scenario.power_plant.basin_init_volumes()
        
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
        
            

